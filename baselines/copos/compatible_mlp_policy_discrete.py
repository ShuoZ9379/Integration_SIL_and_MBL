from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import numpy as np
import gym
from baselines.common.distributions import make_pdtype

class CompatibleMlpPolicyDiscrete(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers):
        assert isinstance(ac_space, gym.spaces.Discrete)

        self.discrete_ob_space = isinstance(ob_space, gym.spaces.Discrete)

        self.pdtype = pdtype = make_pdtype(ac_space)
        self.nact = ac_space.n
        self.varphi_dim = hid_size
        sequence_length = None

        ob_dtype = ob_space.dtype

        self.ob = U.get_placeholder(name="ob", dtype=ob_dtype, shape=(sequence_length,) + ob_space.shape)
        
        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope('vf'):
            if self.discrete_ob_space:
                self.encoded_x = tf.to_float(tf.one_hot(self.ob, ob_space.n))
            else:
                self.encoded_x = tf.clip_by_value((self.ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

            last_out = self.encoded_x
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        with tf.variable_scope('pol'):
            last_out = self.encoded_x
            # Create 'num_hid_layers' hidden layers
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i'%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            self.varphi = last_out

            ### linear-part \theta of the policy using softmax over discrete actions and \phi_{\beta}(s,a)
            self.theta = tf.get_variable(name="theta", shape=[self.varphi_dim, self.nact],
                                                  initializer=tf.random_uniform_initializer(-0.1, 0.1))
            psi_beta_theta = tf.matmul(self.varphi, self.theta)
            #self.psi_beta_theta = psi_beta_theta
            action_prob = tf.nn.softmax(psi_beta_theta)
            log_action_prob = tf.nn.log_softmax(psi_beta_theta)

        self._action_prob = U.function([self.ob], [action_prob])
        self._log_action_prob = U.function([self.ob], [log_action_prob])

        self.pd = pdtype.pdfromflat(psi_beta_theta)

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, self.ob], [ac, self.vpred])

        # action = U.get_placeholder(name="act", dtype=tf.int64, shape=(sequence_length,) + ac_space.shape)
        # logprob = self.pd.logp(action)
        # self._logprob = U.function([action, self.ob], [logprob])

        ### compatible values function approximation

        self.scope = tf.get_variable_scope().name
        # Get non-linear an linear policy parameters (\beta and \theta)
        params =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope + '/pol/')

        # Flat w
        param_len = np.sum([np.prod(p.get_shape().as_list()) for p in params])
        w_var = tf.placeholder(dtype=tf.float32, shape=[param_len])

        # Unflatten w
        param_shapes = list(map(tf.shape, params))
        w_unflat_var = self.unflatten_tensor_variables(w_var, param_shapes)
        
        # w^T * \grad \psi_beta_theta(s)^T
        v = tf.placeholder(dtype=psi_beta_theta.dtype, shape=psi_beta_theta.get_shape(), name="v_in_Rop")
        comp_val_func_approx = self.alternative_Rop(psi_beta_theta, params, w_unflat_var, v)

        self.F_w = U.function([self.ob, w_var, v], comp_val_func_approx)

        # self.scope = tf.get_variable_scope().name
        # # Get non-linear parameters (\beta)
        # params_beta =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope + '/pol/fc')

        # # Flat w
        # beta_len = np.sum([np.prod(p.get_shape().as_list()) for p in params_beta])
        # w_beta_var = tf.placeholder(dtype=tf.float32, shape=[beta_len])

        # # Unflatten w
        # beta_shapes = list(map(tf.shape, params_beta))
        # w_beta_unflat_var = self.unflatten_tensor_variables(w_beta_var, beta_shapes)
        
        # # w^T * \grad \psi_beta_theta(s)^T
        # v = tf.placeholder(dtype=psi_beta_theta.dtype, shape=psi_beta_theta.get_shape(), name="v_in_Rop")
        # features_beta = self.alternative_Rop(psi_beta_theta, params_beta, w_beta_unflat_var, v)

        # self.features_beta = U.function([self.ob, w_beta_var, v], features_beta)


    # def logprob(self, action, ob):
    #     if(isinstance(ob, int) or ob.shape == ()):
    #         tmp_ob = np.zeros((1))
    #         tmp_ob[0] = ob
    #     else:
    #         tmp_ob = ob

    #     if not self.discrete_ob_space:
    #         tmp_ob = ob[None]

    #     if(action.shape == ()):
    #         tmp_action = np.zeros((1))
    #         tmp_action[0] = action
    #     else:
    #         tmp_action = action
    #     #print("logprob(): action="+str(tmp_action)+" ob="+str(ob))
    #     logprob = self._logprob(tmp_action, tmp_ob)
    #     #print(logprob)
    #     return logprob[0]

    def act(self, stochastic, ob):
        if not self.discrete_ob_space:
            tmp_ob = ob[None]
        elif(isinstance(ob, int) or ob.shape == ()):
            tmp_ob = np.zeros((1))
            tmp_ob[0] = ob
        else:
            tmp_ob = ob
        ac1, vpred1 =  self._act(stochastic, tmp_ob)
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []

    def theta_len(self):
        # num (linear) parameters \theta
        return (self.nact * self.varphi_dim)

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

    def get_action_prob(self, obs):
        return self._action_prob(obs)[0]

    def get_log_action_prob(self, obs):
        return self._log_action_prob(obs)[0]

    # def get_varphis(self, obs):
    #     # for debugging
    #     return tf.get_default_session().run(self.varphi, {self.ob: obs})

    # def get_theta(self, obs):
    #     # for debugging
    #     return tf.get_default_session().run(self.theta, {self.ob: obs})

    # def get_psi_beta_theta(self, obs):
    #     # for debugging
    #     return tf.get_default_session().run(self.psi_beta_theta, {self.ob: obs})

    # def get_psi_beta_theta_sym(self):
    #     return self.psi_beta_theta

    def compute_F_w(self, obs, w):
        v0 = np.zeros((obs.shape[0], self.nact))
        return self.F_w(obs, w, v0)[0]
        # w_theta, w_beta = self.split_w(w)
        # #v0 = np.zeros((obs.shape[0], self.varphi_dim))
        # f_beta = self.features_beta(obs,w_beta, v0)[0]
        # w_theta_reshape = np.reshape(w_theta, (self.varphi_dim, self.nact))
        
        # varphi = tf.get_default_session().run(self.varphi, {self.ob: obs})
        # print(varphi.shape)
        # print(w_theta_reshape.shape)
        # q_theta = np.dot(varphi, w_theta_reshape)
        # print(q_theta.shape)
        # print(f_beta.shape)
        # #return np.dot(f_beta,w_theta_reshape)
        # return f_beta + q_theta


    def alternative_Rop(self, f, x, u, v):
        """Alternative implementation of the Rop operation in Theano.
        Please, see
        https://j-towns.github.io/2017/06/12/A-new-trick.html
        https://github.com/renmengye/tensorflow-forward-ad/issues/2
        for an explanation.
        The default value for 'v' should not influence the end result since 'v' is eliminated but
        is needed in some cases to prevent the graph compiler from complaining.
        """
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

