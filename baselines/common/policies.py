import tensorflow as tf
import numpy as np
from baselines.common import tf_util
from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.tf_util import adjust_shape
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.models import get_network_builder

import gym,sys

def dim_reduce(ob):
    if len(ob.shape)==2:
        return ob.flatten()
    else:
        return ob

class PolicyWithValue(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, ob_space, ac_space, env, observations, latent, gaussian_fixed_var, copos, init_std, estimate_q=False, vf_latent=None, sess=None, **tensors):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """
        assert isinstance(ob_space, gym.spaces.Box)
        self.X = observations
        self.state = tf.constant([])
        self.initial_state = None
#        self.cao=tensors['rms']    
        self.__dict__.update(tensors)

        vf_latent = vf_latent if vf_latent is not None else latent

        vf_latent = tf.layers.flatten(vf_latent)
        latent = tf.layers.flatten(latent)
        
        self.varphi=latent
        self.varphi_dim = int(latent.shape[1])
        self.action_dim = ac_space.shape[0]
        self.dist_diagonal = True
        
        # Based on the action space, will select what probability distribution type
        self.pdtype = pdtype = make_pdtype(env.action_space)

        if copos==False:
            self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=0.01) if gaussian_fixed_var else self.pdtype.pdfromlatent_no_fix(latent, init_scale=0.01)
        else:
            if self.dist_diagonal:
                stddev_init = np.ones([1, self.action_dim]) * init_std
                prec_init = 1. / (np.multiply(stddev_init, stddev_init))  # 1 x |a|
                self.prec = tf.get_variable(name="pi/prec", shape=[1, self.action_dim],
                                            initializer=tf.constant_initializer(prec_init))
                kt_init = np.ones([self.varphi_dim, self.action_dim]) * 0.5 / self.varphi_dim
                ktprec_init = kt_init * prec_init
                self.ktprec = tf.get_variable(name="pi/ktprec", shape=[self.varphi_dim, self.action_dim],
                                              initializer=tf.constant_initializer(ktprec_init))
                kt = tf.divide(self.ktprec, self.prec)
                mean = tf.matmul(latent, kt)
                logstd = tf.log(tf.sqrt(1. / self.prec))
            else:
                # Not implemented yet
                raise NotImplementedError
            self.prec_get_flat = tf_util.GetFlat([self.prec])
            self.prec_set_from_flat = tf_util.SetFromFlat([self.prec])
            self.ktprec_get_flat = tf_util.GetFlat([self.ktprec])
            self.ktprec_set_from_flat = tf_util.SetFromFlat([self.ktprec])
            if gaussian_fixed_var==False or isinstance(ac_space, gym.spaces.Box)==False:
                pdparam = tf.layers.dense(latent, pdtype.param_shape()[0], name='final', kernel_initializer=tf_utils.normc_initializer(0.01))
                self.pi=pdparam
            else:
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
                self.pi=mean
                
            self.scope = tf.get_variable_scope().name
            self.pd = pdtype.pdfromflat(pdparam)
            self.state_in = []
            self.state_out = []
            # Get all policy parameters
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope + '/pi')
            # Remove log-linear parameters ktprec and prec to get only non-linear parameters
            del vars[-1]
            del vars[-1]
            beta_params = vars

            # Flat w_beta
            beta_len = np.sum([np.prod(p.get_shape().as_list()) for p in beta_params])
            w_beta_var = tf.placeholder(dtype=tf.float32, shape=[beta_len])

            # Unflatten w_beta
            beta_shapes = list(map(tf.shape, beta_params))
            w_beta_unflat_var = self.unflatten_tensor_variables(w_beta_var, beta_shapes)

            # w_beta^T * \grad_beta \varphi(s)^T
            v = tf.placeholder(dtype=self.varphi.dtype, shape=self.varphi.get_shape(), name="v_in_Rop")
            features_beta = self.alternative_Rop(self.varphi, beta_params, w_beta_unflat_var, v)
            self.features_beta = tf_util.function([self.X, w_beta_var, v], features_beta)

        #self.action = self.pd.sample()[0]
        self.action_sto = self.pd.sample()
        self.action_det = self.pd.mode()

        # Calculate the neg log of our probability
        self.neglogp_sto = self.pd.neglogp(self.action_sto)
        self.neglogp_det = self.pd.neglogp(self.action_det)
        #self.neglogp = self.pd.neglogp(self.action)
        self.sess = sess or tf.get_default_session()

        if estimate_q:
            assert isinstance(env.action_space, gym.spaces.Discrete)
            self.q = fc(vf_latent, 'q', env.action_space.n)
            self.vf = self.q
        else:
            self.vf = fc(vf_latent, 'vf', 1)
            self.vf = self.vf[:,0]

    def get_initial_state(self):
        return []

    def theta_len(self):
        action_dim = self.action_dim
        varphi_dim = self.varphi_dim

        ktprec_len = varphi_dim * action_dim
        if self.dist_diagonal:
            prec_len = action_dim
        else:
            prec_len = action_dim * action_dim

        return (prec_len + ktprec_len)

    def all_to_theta_beta(self, all_params):
        theta_len = self.theta_len()
        theta = all_params[-theta_len:]
        beta = all_params[0:-theta_len]
        return theta, beta

    def theta_beta_to_all(self, theta, beta):
        return np.concatenate([beta, theta])

    def split_w(self, w):
        """
        Split w into w_theta, w_beta
        :param w: [w_beta w_theta]
        :return: w_theta, w_beta
        """
        theta_len = self.theta_len()

        w_beta = w[0:-theta_len]
        w_theta = w[-theta_len:]

        return w_theta, w_beta

    def w2W(self, w_theta):
        """
        Transform w_{theta} to W_aa and W_sa matrices
        :param theta:
        :type theta:
        :return:
        :rtype:
        """
        action_dim = self.action_dim
        varphi_dim = self.varphi_dim

        if self.dist_diagonal:
            prec_len = action_dim
            waa = np.reshape(w_theta[0:prec_len], (action_dim,))
            Waa = np.diag(waa)
        else:
            prec_len = action_dim * action_dim
            Waa = np.reshape(w_theta[0:prec_len], (action_dim,action_dim))

        Wsa = np.reshape(w_theta[prec_len:], (varphi_dim, action_dim))

        return Waa, Wsa

    def get_wa(self, obs, w_beta):
        """
        Compute wa(s)^T = w_beta^T * \grad_beta \varphi_beta(s)^T * K^T * Sigma^-1
        :return: wa(s)^T
        """
        v0 = np.zeros((obs.shape[0], self.varphi_dim))
        f_beta = self.features_beta(obs, w_beta, v0)[0]
        wa = np.dot(f_beta, self.get_ktprec())

        return wa

    def get_varphis(self, obs):
        if len(obs.shape)>len(self.X.shape):
            return tf.get_default_session().run(self.varphi, {self.X: obs.reshape(-1,self.X.shape[-1])})
        else:
            return tf.get_default_session().run(self.varphi, {self.X: obs})
        
    def get_prec_matrix(self):
        if self.dist_diagonal:
            return np.diag(self.get_prec().reshape(-1,))
        else:
            return self.get_prec()

    def is_policy_valid(self, prec, ktprec):
        if np.any(np.abs(ktprec.reshape(-1,1)) > 1e12):
            return False

        if self.dist_diagonal:
            p = prec
        else:
            p = np.linalg.eigvals(prec)

        return np.all(p > 1e-12) and np.all(p < 1e12)

    def is_current_policy_valid(self):
        return self.is_policy_valid(self.get_prec(), self.get_ktprec())

    def is_new_policy_valid(self, eta, omega, w_theta):
        # New policy
        theta_old = self.get_theta()
        theta = (eta * theta_old + w_theta) / (eta + omega)
        prec, ktprec = self.theta2vars(theta)

        return self.is_policy_valid(prec, ktprec)

    def theta2vars(self, theta):
        """
        :param theta:
        :return: [\Sigma^-1, K^T \Sigma^-1],
        """
        action_dim = self.action_dim
        varphi_dim = self.varphi_dim

        if self.dist_diagonal:
            prec_len = action_dim
            prec = np.reshape(theta[0:prec_len], (action_dim,))
            ktprec = np.reshape(theta[prec_len:], (varphi_dim, action_dim))
        else:
            prec_len = action_dim * action_dim
            prec = np.reshape(theta[0:prec_len],
                              (action_dim, action_dim))
            ktprec = np.reshape(theta[prec_len:], (varphi_dim, action_dim))

        return (prec, ktprec)

    def get_ktprec(self):
        """
        :return: K^T \Sigma^-1
        """
        return tf.get_default_session().run(self.ktprec)

    def get_prec(self):
        return tf.get_default_session().run(self.prec)

    def get_sigma(self):
        if self.dist_diagonal:
            return np.diag(1 / self.get_prec().reshape(-1, ))
        else:
            return np.linalg.inv(self.get_prec())

    def get_kt(self):
        return np.dot(self.get_ktprec(), self.get_sigma())

    def get_theta(self):
        """
        :return: \theta
        """
        theta = np.concatenate((self.get_prec().reshape(-1,), self.get_ktprec().reshape(-1,)))
        return theta

    def alternative_Rop(self, f, x, u, v):
        # v = tf.placeholder_with_default(input=v0, dtype=f.dtype, shape=f.get_shape(), name="v_in_Rop")  # dummy variable
        g = tf.gradients(f, x, grad_ys=v)
        return tf.gradients(g, v, grad_ys=u)

    def unflatten_tensor_variables(self, flatarr, shapes):
        arrs = []
        n = 0
        for shape in shapes:
            size = tf.reduce_prod(shape)
            arr = tf.reshape(flatarr[n:n + size], shape)
            arrs.append(arr)
            n += size
        return arrs

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def step(self, observation, stochastic=True, **extra_feed):
        if stochastic:
            a, v, state,neglogp = self._evaluate([self.action_sto, self.vf, self.state,self.neglogp_sto], observation, **extra_feed)
        else:
            a, v, state,neglogp = self._evaluate([self.action_det, self.vf, self.state,self.neglogp_det], observation, **extra_feed)
        if state.size == 0:
            state = None
        return a, v, state, neglogp

    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.vf, ob, *args, **kwargs)

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)

def build_policy(env, policy_network, value_network=None, normalize_observations=False, estimate_q=False, gaussian_fixed_var=True, copos=False, init_std=1.0, **policy_kwargs):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)
       
    def policy_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None):
        ob_space = env.observation_space
        ac_space = env.action_space
        print(observ_placeholder)
        #sys.exit()
        
        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)

        extra_tensors = {}

        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X

        encoded_x = encode_observation(ob_space, encoded_x)

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            policy_latent = policy_network(encoded_x)
            if isinstance(policy_latent, tuple):
                policy_latent, recurrent_tensors = policy_latent

                if recurrent_tensors is not None:
                    # recurrent architecture, need a few more steps
                    nenv = nbatch // nsteps
                    assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
                    policy_latent, recurrent_tensors = policy_network(encoded_x, nenv)
                    extra_tensors.update(recurrent_tensors)


        _v_net = value_network

        if _v_net is None or _v_net == 'shared':
            vf_latent = policy_latent
        else:
            if _v_net == 'copy':
                _v_net = policy_network
            else:
                assert callable(_v_net)

            with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
                # TODO recurrent architectures are not supported with value_network=copy yet
                vf_latent = _v_net(encoded_x)

        policy = PolicyWithValue(
            ob_space=ob_space,
            ac_space=ac_space,
            env=env,
            observations=X,
            latent=policy_latent,
            gaussian_fixed_var=gaussian_fixed_var,
            copos=copos,
            init_std=init_std,
            estimate_q=estimate_q,
            vf_latent=vf_latent,
            sess=sess,
            **extra_tensors
        )
        return policy

    return policy_fn


def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms

