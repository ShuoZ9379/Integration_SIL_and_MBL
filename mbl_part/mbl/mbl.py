import numpy as np
import tensorflow as tf
import copy 
import gym
import scipy.stats as stats

from baselines import logger
from baselines.common import explained_variance, zipsame
import baselines.common.tf_util as U

from mbl.dynamics import ForwardDynamic
from mbl.model_config import make_mlp
from mbl.reward_func import get_reward_done_func

def get_variables(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

class MBL(object):
    def __init__(self, env, env_id, make_model=make_mlp, 
            num_warm_start=int(1e4),            
            init_epochs=20, 
            update_epochs=1, 
            batch_size=512, 
            forward_dynamic=None,
            **kwargs):
        logger.log('MBL args', locals())
       
        self.env = env
        self.reward_done_func = get_reward_done_func(env_id)
        self.ob_space = env.observation_space
        self.ac_space = env.action_space
        
        self.use_self_forward_dynamic = False
        if forward_dynamic is None:
            self.use_self_forward_dynamic = True
            self.forward_dynamic = ForwardDynamic(ob_space=self.ob_space, ac_space=self.ac_space, make_model=make_model, scope='new_forward_dynamic', **kwargs)
            self.old_forward_dynamic = ForwardDynamic(ob_space=self.ob_space, ac_space=self.ac_space, make_model=make_model, scope='old_forward_dynamic', **kwargs)

            self.update_model = U.function([],[], updates=[tf.assign(oldv, newv)
                for (oldv, newv) in zipsame(get_variables("old_forward_dynamic"), get_variables("new_forward_dynamic"))])
            self.restore_model = U.function([],[], updates=[tf.assign(newv, oldv)
                for (oldv, newv) in zipsame(get_variables("old_forward_dynamic"), get_variables("new_forward_dynamic"))])
        else:
            self.forward_dynamic = forward_dynamic
              
        self.num_warm_start = num_warm_start 
        self.init_epochs = init_epochs
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.warm_start_done = False
        self.prev_loss_val = None
        
    def _lookahead(self, pi, state, horizon, num_samples, num_elite, gamma, lamb, use_mean_elites):
        state_shape = state.shape
        state_dim = state_shape[-1]
        act_dim = self.ac_space.shape[0]

        all_state = np.tile(state, num_samples).reshape((num_samples, state_dim))
        all_mask = np.ones(num_samples)
        all_total_reward = np.zeros(num_samples)
        all_last_value = np.zeros(num_samples)
        all_history_value = []
        #all_history_action = []       
        all_history_action = np.empty([horizon, num_samples, act_dim])

        for h in range(horizon):
            # Sample actions
            all_action, all_value = pi(all_state, t=h)
            
            # Forward simulation
            all_state_next = self.forward_dynamic.predict(ob=all_state, ac=all_action) # N x state_dim
            all_reward, all_done = self.reward_done_func(all_state, all_action, all_state_next) # N x 1, N x 1            
            all_reward = gamma**(h) * all_reward        

            # Accumulate the reward
            all_total_reward += (all_reward * all_mask) # N x 1    
            all_masked_value = (all_mask) * all_value  + (np.ones_like(all_mask) - all_mask) * (all_last_value) # Padding with all_last_value if done
            all_last_value = (all_mask) * all_value + (np.ones_like(all_mask) - all_mask) * (all_last_value) # Padding with all_last_value if done
            #all_history_action.append(all_action) # H x N x act_dim       
            all_history_action[h] = all_action
            all_history_value.append(all_masked_value) # H x N x 1

            # Prepare for next iteration
            all_notdone = np.ones_like(all_done) - all_done
            all_mask = (np.ones_like(all_mask) - all_mask) * all_mask + all_mask * all_notdone
            all_state = all_state_next # N x state_dim
            
            if np.all(np.ones_like(all_mask) - all_mask):            
                break

        all_history_value = np.array(all_history_value)
        all_total_reward = all_total_reward + lamb * (gamma)**(horizon) * all_history_value[-1]

        elite_idx = np.argsort(all_total_reward)[::-1][:num_elite]
        best_idx = np.random.choice(elite_idx)
        if use_mean_elites:
            #all_history_action_np = np.stack(all_history_action)
            best_action = np.mean(all_history_action[0][elite_idx], axis=0)
        else:
            best_action = all_history_action[0][best_idx]
        best_reward = all_total_reward[best_idx]

        return best_action, best_reward

    def _add_data(self, ob, ac, ob_next):
        assert self.use_self_forward_dynamic, 'It is invalid to update the external forward dynamics model'
        self.forward_dynamic.append(copy.copy(ob), copy.copy(ac), copy.copy(ob_next))
   
    def update_forward_dynamic(self, require_update, ob_val=None, ac_val=None, ob_next_val=None):
        '''
        Update the forward dynamic model
        '''
        assert self.use_self_forward_dynamic, 'It is invalid to update the external forward dynamics model'
        if not self.is_warm_start_done():
            logger.log('Warm start progress', (self.forward_dynamic.memory.nb_entries / self.num_warm_start))
    
        has_train = False

        # Check if need to update train
        if require_update and self.is_warm_start_done():
            logger.log('Update train')
            self.forward_dynamic.train(self.batch_size, self.update_epochs)
            has_train = True
                   
        # Check if enough for init train
        if self.forward_dynamic.memory.nb_entries >= self.num_warm_start and not self.warm_start_done:
            logger.log('Init train')
            self.forward_dynamic.train(self.batch_size, self.init_epochs)
            self.warm_start_done = True 
            has_train = True
        
        # Check if need to validate
        if has_train:
            if ob_val is not None and ac_val is not None and ob_next_val is not None:               
                logger.log('Validating...')
                loss_val = self.eval_forward_dynamic(ob_val, ac_val, ob_next_val)
                logger.log('Validation loss: {}'.format(loss_val))
                if self.prev_loss_val is not None:                    
                    if self.prev_loss_val < loss_val:
                        logger.log('New model is worse or equal, restore')
                        self.restore_model()
                    else:
                        logger.log('New model is better, update')
                        self.update_model()
                self.prev_loss_val = loss_val                
            else:
                logger.log('Update without validation')
 
    def add_data_batch(self, obs, acs, obs_next):
        '''
        Aggregate the dataset
        '''
        assert self.use_self_forward_dynamic, 'It is invalid to update the external forward dynamics model'
        for ob, ac, ob_next in zip(obs, acs, obs_next):
            self._add_data(ob, ac, ob_next)

    def eval_forward_dynamic(self, obs, acs, obs_next):
        return self.forward_dynamic.eval(obs, acs, obs_next)
        
    def is_warm_start_done(self):
        return self.warm_start_done
            
    def step(self, ob, pi, horizon, num_samples, num_elites, gamma, lamb, use_mean_elites):
        ac, rew = self._lookahead(pi=pi, state=ob, horizon=horizon, num_samples=num_samples, num_elite=num_elites, gamma=gamma, lamb=lamb, use_mean_elites=use_mean_elites)
        return ac, rew 
        
class MBLCEM(MBL):
    def __init__(self, env, env_id, horizon, make_model=make_mlp, 
            num_warm_start=int(1e4),            
            init_epochs=20, 
            update_epochs=1, 
            batch_size=512, 
            forward_dynamic=None,
            **kwargs):
        super(MBLCEM, self).__init__(env, env_id, make_model, num_warm_start, init_epochs, update_epochs, batch_size, forward_dynamic, **kwargs)
        assert hasattr(self.ac_space, 'low') and hasattr(self.ac_space, 'high') and hasattr(self.ac_space, 'shape')

        self.horizon = horizon        
        self.ac_ub, self.ac_lb = self.ac_space.high, self.ac_space.low
        self.ac_dim = self.ac_space.shape[-1]
        self.reset()

    def reset(self):
        ac_mean = ((self.ac_lb + self.ac_ub) / 2)
        ac_var = np.square((self.ac_ub - self.ac_lb) / 16)
        
        self.acseq_mean = np.random.uniform(low=self.ac_lb, high=self.ac_ub, size=(self.horizon, self.ac_dim))        
        self.acseq_var = np.tile(np.square((self.ac_ub - self.ac_lb) / 2), (self.horizon, 1))

        #self.acseq_mean = np.tile(ac_mean, (self.horizon, 1))
        #self.acseq_var = np.tile(ac_var, (self.horizon, 1))

    def _lookahead(self, pi, vf, state, num_samples, num_iters, num_elite, gamma, lamb, use_mean_elites):
        horizon = self.horizon

        state_shape = state.shape
        state_dim = state_shape[-1]
        action_dim = self.ac_space.shape[-1]
        actionseq_dim = (horizon, action_dim)
        
        # Constrained implementation
        X = stats.truncnorm(-2, 2, loc=np.zeros_like(self.acseq_mean), scale=np.ones_like(self.acseq_var))        
        acseq_mean = self.acseq_mean.copy()
        acseq_var = self.acseq_var.copy()

        for i in range(num_iters):
            all_state = np.tile(state, num_samples).reshape((num_samples, state_dim))
            all_mask = np.ones(num_samples)
            all_total_reward = np.zeros(num_samples)
            all_last_value = np.zeros(num_samples)
            all_history_value = []
            all_history_action = []        

            # Constrained implementation
            lb_dist, ub_dist = acseq_mean - self.ac_lb, self.ac_ub - acseq_mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), acseq_var)
            all_actionseq = X.rvs(size=[num_samples, self.horizon, self.ac_dim]) * np.sqrt(constrained_var) + acseq_mean
                      
            for h in range(horizon):
                # Sample actions  
                all_action = all_actionseq[:, h, ...]
                all_value = vf(all_state)
                
                # Forward simulation
                all_state_next = self.forward_dynamic.predict(ob=all_state, ac=all_action) # N x state_dim
                all_reward, all_done = self.reward_done_func(all_state, all_action, all_state_next) # N x 1, N x 1          
                all_reward = gamma**(h) * all_reward        

                # Accumulate the reward
                all_total_reward += (all_reward * all_mask) # N x 1    
                all_masked_value = (all_mask) * all_value  + (np.ones_like(all_mask) - all_mask) * (all_last_value) # Padding with all_last_value if done
                all_last_value = (all_mask) * all_value + (np.ones_like(all_mask) - all_mask) * (all_last_value) # Padding with all_last_value if done
                all_history_action.append(all_action) # H x N x act_dim       
                all_history_value.append(all_masked_value) # H x N x 1

                # Prepare for next iteration
                all_notdone = np.ones_like(all_done) - all_done
                all_mask = (np.ones_like(all_mask) - all_mask) * all_mask + all_mask * all_notdone
                all_state = all_state_next # N x state_dim
                
                if np.all(all_done):            
                    break

            all_history_value_np = np.array(all_history_value)
            all_total_reward = all_total_reward + lamb * gamma**(horizon) * all_history_value_np[-1]
                    
            elite_idx = np.argsort(all_total_reward)[::-1][:num_elite]
            acseq_mean = 0.1 * acseq_mean + (0.9) * np.mean(all_actionseq[elite_idx], axis=0)
            acseq_var = 0.1 * acseq_var + (0.9) * np.var(all_actionseq[elite_idx], axis=0)
            #best_idx = np.random.choice(elite_idx)
            best_idx = np.argmax(all_total_reward)
            best_action = all_history_action[0][best_idx]
            best_reward = all_total_reward[best_idx]
        self.acseq_mean = acseq_mean

        return best_action, best_reward

    def step(self, ob, pi, vf, num_samples, num_iters, num_elites, gamma, lamb, use_mean_elites=False):
        ac, rew = self._lookahead(pi=pi, vf=vf, state=ob, num_samples=num_samples, num_iters=num_iters, num_elite=num_elites, gamma=gamma, lamb=lamb, use_mean_elites=use_mean_elites)
        return ac, rew

class MBLMPPI(MBLCEM):
    def _lookahead(self, pi, vf, state, num_samples, num_iters, num_elite, gamma, lamb, use_mean_elites):
        horizon = self.horizon

        state_shape = state.shape
        state_dim = state_shape[-1]
        action_dim = self.ac_space.shape[-1]
        actionseq_dim = (horizon, action_dim)

        # Constrained implementation
        X = stats.truncnorm(-2, 2, loc=np.zeros_like(self.acseq_mean), scale=np.ones_like(self.acseq_var))        

        for i in range(num_iters):
            all_state = np.tile(state, num_samples).reshape((num_samples, state_dim))
            all_mask = np.ones(num_samples)
            all_total_reward = np.zeros(num_samples)
            all_last_value = np.zeros(num_samples)
            all_history_value = []
            all_history_action = []               

            # Constrained implementation
            lb_dist, ub_dist = self.acseq_mean - self.ac_lb, self.ac_ub - self.acseq_mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), self.acseq_var)
            all_actionseq = X.rvs(size=[num_samples, self.horizon, self.ac_dim]) * np.sqrt(constrained_var) + self.acseq_mean
                      
            for h in range(horizon):
                # Sample actions  
                all_action = all_actionseq[:, h, ...]
                all_value = vf(all_state)
                
                # Forward simulation
                all_state_next = self.forward_dynamic.predict(ob=all_state, ac=all_action) # N x state_dim
                all_reward, all_done = self.reward_done_func(all_state, all_action, all_state_next) # N x 1, N x 1            
                all_reward = gamma**(h) * all_reward        

                # Accumulate the reward
                all_total_reward += (all_reward * all_mask) # N x 1    
                all_masked_value = (all_mask) * all_value  + (np.ones_like(all_mask) - all_mask) * (all_last_value) # Padding with all_last_value if done
                all_last_value = (all_mask) * all_value + (np.ones_like(all_mask) - all_mask) * (all_last_value) # Padding with all_last_value if done
                all_history_action.append(all_action) # H x N x act_dim       
                all_history_value.append(all_masked_value) # H x N x 1

                # Prepare for next iteration
                all_mask = np.ones_like(all_done) - all_done
                all_state = all_state_next # N x state_dim
                
                if np.all(all_done):            
                    break

            all_history_value_np = np.array(all_history_value)
            all_total_reward = all_total_reward + lamb * gamma**(horizon) * all_history_value_np[-1]
            
            mppi_lamb = 5.0
            
            all_epislon = all_actionseq - self.acseq_mean # (N x H x A)
            all_stk = -all_total_reward # (N x 1)
            beta = np.min(all_stk) # (1)
            norm = np.sum(np.exp((1. / mppi_lamb) * (all_stk - beta))) # (1)
            all_wepislon = (1.0 / norm) * np.exp( (1. / mppi_lamb) * (all_stk - beta)) # (N x 1)
            all_wepislon = np.tile(all_wepislon, (self.horizon, 1)).transpose((1, 0))[:, :, np.newaxis]
            self.acseq_mean += np.sum(all_wepislon * all_epislon, axis=0) # (N x H x A)
        
        ac = self.acseq_mean[0].copy()
        self.acseq_mean[:-1, ...] = self.acseq_mean[1:, ...]
        self.acseq_mean[-1, ...] = (self.ac_ub + self.ac_lb) / 2
        
        return ac, -beta
