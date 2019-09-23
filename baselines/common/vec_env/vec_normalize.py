from . import VecEnvWrapper
import numpy as np
import sys

class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, ob=True, ret=False, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8, is_training=True, use_tf=False):
        VecEnvWrapper.__init__(self, venv)
        if use_tf:
            from baselines.common.running_mean_std import TfRunningMeanStd
            self.ob_rms = TfRunningMeanStd(shape=self.observation_space.shape, scope='ob_rms') if ob else None
            self.ret_rms = TfRunningMeanStd(shape=(), scope='ret_rms') if ret else None
        else:
            from baselines.common.running_mean_std import RunningMeanStd
            self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
            self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.i=0

        self.is_training=is_training

    def step_wait(self):
        self.i=self.i+1
        obs, rews, news, infos = self.venv.step_wait()
        #if self.i==3:
            #print (self.ob_rms.var)
            #print(obs)
            #sys.exit()
        self.raw_reward = rews
        self.raw_obs = obs
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        rews = self._rewfilt(rews)
        #if self.ret_rms:
        #    self.ret_rms.update(self.ret)
        #    rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos
    
    def _rewfilt(self, rews):    
        if self.ret_rms:
            if self.is_training: self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        return rews
    
    def _rewdefilt(self, rews):
        if self.ret_rms:
            return rews * np.sqrt(self.ret_rms.var + self.epsilon)
        else:
            return rews

    '''def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs'''
    def _obfilt(self, obs):
        if self.ob_rms:
            if self.is_training: self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs
    
    def _obdefilt(self, obs):
        if self.ob_rms:
            return (obs * np.sqrt(self.ob_rms.var + self.epsilon)) + self.ob_rms.mean
        else:
            return obs
    def process_reward(self, rews):
        if self.ret_rms:
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        return rews
    def process_obs(self, obs):
        if self.ob_rms: 
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
        return obs
    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        self.raw_obs = obs
        return self._obfilt(obs)
