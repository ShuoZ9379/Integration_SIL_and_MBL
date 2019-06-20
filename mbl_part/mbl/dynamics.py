import os
import pickle

import numpy as np
import tensorflow as tf

import gym

from baselines import logger
from baselines.common.running_mean_std import RunningMeanStd

from mbl.model import NN
from mbl.memory import Memory
from mbl.util.util import normalize, denormalize

class ForwardDynamic(object):
    def __init__(self, ob_space, ac_space, make_model, 
            normalize_obs=False,             
            memory_size=int(1e5),
            scope='forward_dynamic', 
            **kwargs):
        logger.log('Forward dynamic args', locals())
        
        self.ob_space = ob_space
        self.ac_space = ac_space

        self.num_ob = int(np.prod(self.ob_space.shape))
        if isinstance(self.ac_space, gym.spaces.Discrete):
            self.num_ac = int(self.ac_space.n)
            self.use_one_hot = True
        elif isinstance(self.ac_space, gym.spaces.Box):
            self.num_ac = int(np.prod(self.ac_space.shape))
            self.use_one_hot = False
        else:
            raise NotImplemented('Not support')

        self.input_shape = (self.num_ob + self.num_ac,)
        self.output_shape = (self.num_ob,)

        # TODO: add PNN, for now DNN
        with tf.variable_scope(scope) as _:
            self.model = NN(num_input=self.input_shape[0], num_output=self.output_shape[0], make_model=make_model, **kwargs)
        
        self.memory = Memory(limit=memory_size, action_shape=(self.num_ac,), observation_shape=(self.num_ob,))
        self.normalize_obs = normalize_obs
        if self.normalize_obs:
            self.ob_rms = RunningMeanStd(shape=(self.num_ob,))

    def _preprocess_ob(self, ob):
        '''
        Reshape the observation
        '''
        ob = np.reshape(ob, (-1, self.num_ob,))        
        return ob

    def _preprocess_ac(self, ac):
        '''
        Reshape the action
        '''
        if self.use_one_hot:
            if not isinstance(ac, np.ndarray): ac = np.array([ac,]) 
            tmp = np.zeros((len(ac), self.num_ac))       
            tmp[np.arange(len(ac)).astype(np.int), ac.astype(np.int)] = 1
            return tmp
        else:
            return ac

    def _preprocess_model_inputs_targets(self, ob_proc, ac_proc, ob_next_proc):       
        if self.normalize_obs:
            ob_proc_normalize, ob_next_proc_normalize = normalize(ob_proc, self.ob_rms), normalize(ob_next_proc, self.ob_rms)        
            ob_diff_proc_normalize = ob_next_proc_normalize - ob_proc_normalize
            inputs = np.concatenate([ob_proc_normalize, ac_proc], axis=1)
            targets = ob_diff_proc_normalize      
        else:
            ob_diff_proc = ob_next_proc - ob_proc       
            inputs = np.concatenate([ob_proc, ac_proc], axis=1)      
            targets = ob_diff_proc

        return inputs, targets

    def _update_rms(self, ob_proc, ac_proc):
        if self.normalize_obs:
            self.ob_rms.update(ob_proc)

    def _train_epochs(self, batch_size, n_epochs):
        ob_proc_train, ac_proc_train, ob_next_proc_train = self.memory.data(size=self.memory.nb_entries)
        
        n = self.memory.nb_entries
        for epoch in range(n_epochs):
            epoch_losses = []
            idxes = np.arange(n)
            np.random.shuffle(idxes)                    
            for start_idx in range(0, n, batch_size):
                end_idx = min(start_idx + batch_size, n)
                batch = idxes[start_idx:end_idx]
                ob_proc_batch, ac_proc_batch, ob_next_proc_batch = ob_proc_train[batch], ac_proc_train[batch], ob_next_proc_train[batch]
                loss = self._train_batch(ob_proc_batch, ac_proc_batch, ob_next_proc_batch)        
                epoch_losses.append(loss)
            logger.log('# batch per epoch', len(epoch_losses))
            logger.log('Epoch', epoch, np.mean(epoch_losses))

    def _train_batch(self, ob_proc, ac_proc, ob_next_proc):
        inputs, targets = self._preprocess_model_inputs_targets(ob_proc, ac_proc, ob_next_proc)
        return self.model.train(inputs, targets)

    def _eval_batch(self, ob_proc, ac_proc, ob_next_proc):
        inputs, targets = self._preprocess_model_inputs_targets(ob_proc, ac_proc, ob_next_proc)
        return self.model.eval(inputs, targets)        

    def train(self, batch_size, n_epochs):
        '''
        Sample batch_size data from memory and train n_epochs
        '''
        self._train_epochs(batch_size=batch_size, n_epochs=n_epochs)

    def append(self, ob, ac, ob_next):        
        '''
        Aggregate the dataset with new (ob, ac, ob_next)
        '''
        ob_proc, ac_proc, ob_next_proc = self._preprocess_ob(ob), self._preprocess_ac(ac), self._preprocess_ob(ob_next)

        self.memory.append(ob_proc, ac_proc, ob_next_proc)
            
        if self.normalize_obs:
            self._update_rms(ob_proc, ac_proc)
            
    def predict(self, ob, ac):
        '''
        Predict ob_next given (ob, ac)
        '''
        ob_proc, ac_proc = self._preprocess_ob(ob), self._preprocess_ac(ac)
        
        if self.normalize_obs:
            ob_proc_normalize = normalize(ob_proc, self.ob_rms)       
            inputs = np.concatenate([ob_proc_normalize, ac_proc], axis=1)
            ob_diff_pred_normalize = self.model.predict(inputs)
            ob_next_pred_normalize = ob_proc_normalize + ob_diff_pred_normalize
            ob_next_pred = denormalize(ob_next_pred_normalize, self.ob_rms)
        else:
            inputs = np.concatenate([ob_proc, ac_proc], axis=1)           
            ob_diff_pred = self.model.predict(inputs)
            ob_next_pred = ob_proc + ob_diff_pred
           
        return ob_next_pred       

    def eval(self, ob, ac, ob_next):      
        ob_proc, ac_proc, ob_next_proc = self._preprocess_ob(ob), self._preprocess_ac(ac), self._preprocess_ob(ob_next)
        return self._eval_batch(ob_proc, ac_proc, ob_next_proc)
