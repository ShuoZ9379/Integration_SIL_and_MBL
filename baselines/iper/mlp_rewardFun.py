from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
import numpy as np

class MlpRewardFun(object):
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, use_actions):
        assert isinstance(ob_space, gym.spaces.Box)
        self.use_actions = use_actions
        sequence_length = None

        if use_actions:
            inp_shape = (ob_space.shape[0] + ac_space.shape[0],)
        else:
            inp_shape = ob_space.shape
        rew_input = U.get_placeholder(name="rew_input", dtype=tf.float32, shape=[sequence_length] + list(inp_shape))

        with tf.variable_scope("inputfilter"):
            self.inp_rms = RunningMeanStd(shape=inp_shape)

        with tf.variable_scope('rew'):
            input_clipped = tf.clip_by_value((rew_input - self.inp_rms.mean) / self.inp_rms.std, -5.0, 5.0)
            last_out = input_clipped
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            self.reward = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        self._rew = U.function([rew_input], [self.reward])

    def getReward(self, ob, ac):
        if self.use_actions:
            rew = self._rew(np.hstack((ob,ac))[None])
        else:
            rew = self._rew(ob[None])
        return rew

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

