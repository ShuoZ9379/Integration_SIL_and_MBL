import os,sys, time,pickle, copy
import numpy as np, tensorflow as tf
import os.path as osp
from baselines import logger
from collections import deque
import baselines.common.tf_util as U
from baselines.common import explained_variance, set_global_seeds, colorize
from baselines.common.policies import build_policy
from contextlib import contextmanager
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
# MBL
from mbl.mbl import MBL, MBLCEM, MBLMPPI
from mbl.exp_util import eval_policy, Policy
from mbl.util.util import load_extracted_val_data as load_val_data
from mbl.util.util import to_onehot
from mbl.model_config import get_make_mlp_model
#from plot import plot 
#from visdom import Visdom
from multiprocessing.dummy import Pool
import multiprocessing as mp
from runner import Runner
from model_novec import Model

def constfn(val):
    def f(_):
        return val
    return f

def learn(*, network,
        env, eval_env, make_eval_env, env_id,
        total_timesteps, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,
        vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
        log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
        sil_update=10, sil_value=0.01, sil_alpha=0.6, sil_beta=0.1, sil_loss=0.1,

        # MBL
        # For train mbl
        mbl_train_freq=5,
        # For eval
        num_eval_episodes=5,
        eval_freq=5,
        vis_eval=False,
#        eval_targs=('mbmf',),
        eval_targs=('mf',),
        quant=2,

        # For mbl.step
        #num_samples=(1500,),
        num_samples=(1,),
        horizon=(2,),
        #horizon=(2,1),
        #num_elites=(10,),
        num_elites=(1,),
        mbl_lamb=(1.0,),
        mbl_gamma=0.99,
        #mbl_sh=1, # Number of step for stochastic sampling
        mbl_sh=10000,
        #vf_lookahead=-1,
        #use_max_vf=False,
        reset_per_step=(0,),

        # For get_model
        num_fc=2,
        num_fwd_hidden=500,
        use_layer_norm=False,        

        # For MBL
        num_warm_start=int(1e4),            
        init_epochs=10, 
        update_epochs=5, 
        batch_size=512, 
        update_with_validation=False, 
        use_mean_elites=1,
        use_ent_adjust=0,
        adj_std_scale=0.5,

        # For data loading
        validation_set_path=None, 

        # For data collect
        collect_val_data=False,

        # For traj collect
        traj_collect='mf',      
        
        # For profile
        measure_time=True,
        eval_val_err=False,
        measure_rew=True,
        save_interval=0, load_path=None, model_fn=None, update_fn=None, init_fn=None, mpi_rank_weight=1, comm=None, **network_kwargs):
    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    noptepochs: int                   number of training epochs per update

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.



    '''
    if not isinstance(num_samples, tuple): num_samples = (num_samples,)
    if not isinstance(horizon, tuple): horizon = (horizon,)
    if not isinstance(num_elites, tuple): num_elites = (num_elites,)
    if not isinstance(mbl_lamb, tuple): mbl_lamb = (mbl_lamb,)
    if not isinstance(reset_per_step, tuple): reset_per_step = (reset_per_step,)
    if validation_set_path is None: 
        if collect_val_data: validation_set_path = os.path.join(logger.get_dir(), 'val.pkl')
        else: validation_set_path = os.path.join('dataset', '{}-val.pkl'.format(env_id))
    if eval_val_err:
        eval_val_err_path = os.path.join('dataset', '{}-combine-val.pkl'.format(env_id))
    logger.log(locals())
    logger.log('MBL_SH', mbl_sh)
    logger.log('Traj_collect', traj_collect)
    
    if MPI is not None:
        nworkers = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        nworkers = 1
        rank = 0  
    cpus_per_worker = 1
    U.get_session(config=tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=cpus_per_worker,
            intra_op_parallelism_threads=cpus_per_worker
    ))

    set_global_seeds(seed)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)
    
    policy = build_policy(env, network, **network_kwargs)
    np.set_printoptions(precision=3)
    # Get the nb of env
    nenvs = env.num_envs
    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        model_fn = Model

    make_model = lambda: Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                          nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                          max_grad_norm=max_grad_norm,
                          sil_update=sil_update,
                          fn_reward=None, fn_obs=None,
                          sil_value=sil_value, 
                          sil_alpha=sil_alpha, 
                          sil_beta=sil_beta,
                          sil_loss=sil_loss,
                          comm=comm, mpi_rank_weight=mpi_rank_weight,
                          ppo=True,prev_pi=None)
    model=make_model()
    pi=model.sil_model
    
    if load_path is not None:
        model.load(load_path)

    # MBL
    # ---------------------------------------
    #viz = Visdom(env=env_id) 
    win = None
    eval_targs = list(eval_targs)
    logger.log(eval_targs)

    make_model_f = get_make_mlp_model(num_fc=num_fc, num_fwd_hidden=num_fwd_hidden, layer_norm=use_layer_norm)
    mbl = MBL(env=eval_env, env_id=env_id, make_model=make_model_f,
            num_warm_start=num_warm_start,            
            init_epochs=init_epochs, 
            update_epochs=update_epochs, 
            batch_size=batch_size, 
            **network_kwargs)

    val_dataset = {'ob': None, 'ac': None, 'ob_next': None}
    if update_with_validation:
        logger.log('Update with validation')
        val_dataset = load_val_data(validation_set_path)
    if eval_val_err:
        logger.log('Log val error')
        eval_val_dataset = load_val_data(eval_val_err_path)       
    if collect_val_data:
        logger.log('Collect validation data')
        val_dataset_collect = [] 

    def _mf_pi(ob, t=None):
        stochastic = True
        ac, vpred, _, _ = pi.step(ob, stochastic=stochastic)
        return ac, vpred   
    def _mf_det_pi(ob, t=None):
        #ac, vpred, _, _ = pi.step(ob, stochastic=False)
        ac, vpred = pi._evaluate([pi.pd.mode(), pi.vf], ob)        
        return ac, vpred
    def _mf_ent_pi(ob, t=None):
        mean, std, vpred = pi._evaluate([pi.pd.mode(), pi.pd.std, pi.vf], ob)
        ac = np.random.normal(mean, std * adj_std_scale, size=mean.shape)
        return ac, vpred
################### use_ent_adjust======> adj_std_scale????????pi action sample
    def _mbmf_inner_pi(ob, t=0):
        if use_ent_adjust:
            return _mf_ent_pi(ob)
        else:
            #return _mf_pi(ob)
            if t < mbl_sh: return _mf_pi(ob)        
            else: return _mf_det_pi(ob)

   # ---------------------------------------   
 
    # Run multiple configuration once
    all_eval_descs = []
    def make_mbmf_pi(n, h, e, l):
        def _mbmf_pi(ob):                        
            ac, rew = mbl.step(ob=ob, pi=_mbmf_inner_pi, horizon=h, num_samples=n, num_elites=e, gamma=mbl_gamma, lamb=l, use_mean_elites=use_mean_elites) 
            return ac[None], rew
        return Policy(step=_mbmf_pi, reset=None)

    for n in num_samples:
        for h in horizon:
            for l in mbl_lamb:
                for e in num_elites:                     
                    if 'mbmf' in eval_targs: all_eval_descs.append(('MeanRew', 'MBL_PPO_SIL', make_mbmf_pi(n, h, e, l)))
                    #if 'mbmf' in eval_targs: all_eval_descs.append(('MeanRew-n-{}-h-{}-e-{}-l-{}-sh-{}-me-{}'.format(n, h, e, l, mbl_sh, use_mean_elites), 'MBL_TRPO-n-{}-h-{}-e-{}-l-{}-sh-{}-me-{}'.format(n, h, e, l, mbl_sh, use_mean_elites), make_mbmf_pi(n, h, e, l)))                   
    if 'mf' in eval_targs: all_eval_descs.append(('MeanRew', 'PPO_SIL', Policy(step=_mf_pi, reset=None)))
   
    logger.log('List of evaluation targets')
    for it in all_eval_descs:
        logger.log(it[0])    

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        else:
            yield

    pool = Pool(mp.cpu_count())
    warm_start_done = False
    U.initialize()
    if load_path is not None:
        pi.load(load_path)

    # Instantiate the runner object
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    epinfobuf = deque(maxlen=40)
    if init_fn is not None: init_fn()

    if traj_collect == 'mf':
        obs= runner.run()[0]
    
    # Start total timer
    tfirststart = time.perf_counter()

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        # Start timer
        if hasattr(model.train_model, "ret_rms"):
            model.train_model.ret_rms.update(returns)
        if hasattr(model.train_model, "rms"):
            model.train_model.rms.update(obs)
        tstart = time.perf_counter()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)

        if update % log_interval == 0 and is_mpi_root: logger.info('Stepping environment...')

        # Get minibatch
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
        
        # Val data collection
        if collect_val_data:
            for ob_, ac_, ob_next_ in zip(obs[:-1, 0, ...], actions[:-1, ...], obs[1:, 0, ...]):            
                val_dataset_collect.append((copy.copy(ob_), copy.copy(ac_), copy.copy(ob_next_)))
        # -----------------------------
        # MBL update
        else:
            ob_mbl, ac_mbl = obs.copy(), actions.copy()
        
            mbl.add_data_batch(ob_mbl[:-1, ...], ac_mbl[:-1, ...], ob_mbl[1:, ...])
            mbl.update_forward_dynamic(require_update=(update-1) % mbl_train_freq == 0, 
                    ob_val=val_dataset['ob'], ac_val=val_dataset['ac'], ob_next_val=val_dataset['ob_next'])            
        # -----------------------------
        
        if update % log_interval == 0 and is_mpi_root: logger.info('Done.')

        epinfobuf.extend(epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        if states is None: # nonrecurrent version
            # Index of each element of batch_size
            # Create the indices array
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
            l_loss, sil_adv, sil_samples, sil_nlogp = model.sil_train(lrnow)
            
        else: # recurrent version
            print("caole")
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        # End timer
        tnow = time.perf_counter()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))

        if update_fn is not None:
            update_fn(update)

        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("misc/serial_timesteps", update*nsteps)
            logger.logkv("misc/nupdates", update)
            logger.logkv("misc/total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("misc/explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv("AverageReturn", safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('misc/time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv('loss/' + lossname, lossval)
            if sil_update > 0:
                logger.logkv("sil_samples", sil_samples)
                    
            if rank==0:
                # MBL evaluation
                if not collect_val_data:
                    #set_global_seeds(seed)
                    default_sess = tf.get_default_session()
                    def multithread_eval_policy(env_, pi_, num_episodes_, vis_eval_,seed):
                        with default_sess.as_default():
                            if hasattr(env, 'ob_rms') and hasattr(env_, 'ob_rms'):
                                env_.ob_rms = env.ob_rms 
                            res = eval_policy(env_, pi_, num_episodes_, vis_eval_, seed, measure_time, measure_rew) 
                            
                            try:
                                env_.close()
                            except:
                                pass
                        return res

                    if mbl.is_warm_start_done() and update % eval_freq == 0:
                        warm_start_done = mbl.is_warm_start_done()
                        if num_eval_episodes > 0 :
                            targs_names = {}
                            with timed('eval'):
                                num_descs = len(all_eval_descs)
                                list_field_names = [e[0] for e in all_eval_descs]
                                list_legend_names = [e[1] for e in all_eval_descs]
                                list_pis = [e[2] for e in all_eval_descs]                    
                                list_eval_envs = [make_eval_env() for _ in range(num_descs)]
                                list_seed= [seed for _ in range(num_descs)]
                                list_num_eval_episodes = [num_eval_episodes for _ in range(num_descs)]
                                print(list_field_names)
                                print(list_legend_names)
                                
                                list_vis_eval = [vis_eval for _ in range(num_descs)]

                                for i in range(num_descs):
                                    field_name, legend_name=list_field_names[i], list_legend_names[i],
                                    
                                    res= multithread_eval_policy(list_eval_envs[i], list_pis[i], list_num_eval_episodes[i], list_vis_eval[i], seed)
                                #eval_results = pool.starmap(multithread_eval_policy, zip(list_eval_envs, list_pis, list_num_eval_episodes, list_vis_eval,list_seed))
                                
                                #for field_name, legend_name, res in zip(list_field_names, list_legend_names, eval_results):
                                    perf, elapsed_time, eval_rew = res
                                    logger.logkv(field_name, perf)                    
                                    if measure_time: logger.logkv('Time-%s' % (field_name), elapsed_time)
                                    if measure_rew: logger.logkv('SimRew-%s' % (field_name), eval_rew)
                                    targs_names[field_name] = legend_name
        
                        if eval_val_err:
                            fwd_dynamics_err = mbl.eval_forward_dynamic(obs=eval_val_dataset['ob'], acs=eval_val_dataset['ac'], obs_next=eval_val_dataset['ob_next'])        
                            logger.logkv('FwdValError', fwd_dynamics_err)

                        #logger.dump_tabular()
                        logger.dumpkvs()
                        #print(logger.get_dir())
                        #print(targs_names)
                        #if num_eval_episodes > 0:
#                            win = plot(viz, win, logger.get_dir(), targs_names=targs_names, quant=quant, opt='best')
                    #else: logger.dumpkvs()
                # -----------
            yield pi
            
        if collect_val_data:
            with open(validation_set_path, 'wb') as f:
                pickle.dump(val_dataset_collect, f)
            logger.log('Save {} validation data'.format(len(val_dataset_collect)))
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and is_mpi_root:
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
        

    return model
# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def get_variables(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

def get_trainable_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

def get_vf_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'vf' in v.name[len(scope):].split('/')]

def get_pi_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'pi' in v.name[len(scope):].split('/')]



