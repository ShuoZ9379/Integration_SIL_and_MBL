import scipy.optimize

# import numpy as np
import autograd.numpy as np  # Thinly-wrapped numpy
#from autograd import grad

import tensorflow as tf
from baselines import logger
import baselines.common.tf_util as U

class EtaOmegaOptimizerDiscrete(object):
    """
    Finds eta and omega Lagrange multipliers for discrete actions.
    """

    def __init__(self, beta, epsilon, init_eta, init_omega):
        self.init_eta_omega(beta, epsilon, init_eta, init_omega)

    def optimize(self, F_w, log_action_prob, batchsize, entropy, eta=None):

        f_dual = self.opt_info['f_dual']
        f_dual_grad = self.opt_info['f_dual_grad']

        # Set BFGS eval function
        def eval_dual(input):
            param_eta = input[0]
            param_omega = input[1]
            val = f_dual(*([F_w, log_action_prob] + [param_eta, param_omega, batchsize, entropy]))
            return val.astype(np.float64)

        # Set BFGS gradient eval function
        def eval_dual_grad(input):
            param_eta = input[0]
            param_omega = input[1]
            grad = f_dual_grad(*([F_w, log_action_prob] + [param_eta, param_omega, batchsize, entropy]))
            return np.asarray(grad)

        if eta is not None:
            param_eta = eta
        else:
            param_eta = self.param_eta

        eta_before = param_eta
        omega_before = self.param_omega
        dual_before = eval_dual([eta_before, omega_before])

        x0 = [param_eta, self.param_omega]

        def fx(x):
            eta, omega = x # eta: Lagrange variable of KL constraint, omega: of the entropy constraint
            error_return_val = 1e6, np.array([0., 0.])
            if (eta + omega < 0) or (eta == 0):
                return error_return_val
            #print("eta: "+str(eta)+"\t omega: "+str(omega))
            #print("dual: "+ str(eval_dual(x)))
            #print("dual_grad: "+ str(eval_dual_grad(x)))

            return eval_dual(x), eval_dual_grad(x)                         # SLSQP
            #return np.float64(eval_dual(x)), np.float64(eval_dual_grad(x)) # L-BFGS-B expects double floats

        logger.log('optimizing dual')

        if eta is None:
            #omega_lower = None
            #omega_lower = 1e-12
            res = scipy.optimize.minimize(fx, x0, method='SLSQP', jac=True,
            #res = scipy.optimize.minimize(fx, x0, method='L-BFGS-B', jac=True,
                                            bounds=((1e-12, None), (1e-12, None)), options={'ftol': 1e-12})
            # Make sure that eta > omega
            if res.x[1] < 0 and -res.x[1] > res.x[0]:
                res.x[1] = -res.x[0] + 1e-6
        else:
            # Fixed eta: make sure that eta > omega
            #omega_lower = np.max([-(eta - 1e-3) + 1e-6, -100])
            #omega_lower = 1e-12
            res = scipy.optimize.minimize(fx, x0, method='SLSQP', jac=True,
            #res = scipy.optimize.minimize(fx, x0, method='L-BFGS-B', jac=True,
                                            bounds=((eta - 1e-3, eta + 1e-3), (1e-12, None)), 
                                            options={'ftol': 1e-16})
        

        if self.beta == 0:
            res.x[1] = 0

        logger.log("dual optimized, eta: " + str(res.x[0]) + ", omega: " + str(res.x[1]))
        return res.x[0], res.x[1]

    def init_eta_omega(self, beta, epsilon, init_eta, init_omega):
        # Here we define the symbolic function for the dual and the gradient

        self.beta = beta
        self.epsilon = epsilon

        # Init dual param values
        self.param_eta = init_eta
        self.param_omega = init_omega

        self.param_eta_non_lin = init_eta
        self.param_omega_non_lin = init_omega

        param_eta = tf.placeholder(dtype=tf.float32, shape=[], name="param_eta")
        param_omega = tf.placeholder(dtype=tf.float32, shape=[], name="param_omega")
        batchsize = tf.placeholder(dtype=tf.float32, shape=[], name="batchsize")
        entropy = tf.placeholder(dtype=tf.float32, shape=[], name="entropy")

        F_w = tf.placeholder(dtype=tf.float32, shape=[None, None], name="F_w")
        log_action_prob = tf.placeholder(dtype=tf.float32, shape=[None, None], name="log_action_prob")

        # Symbolic function for the dual
        dual = param_eta * self.epsilon + param_omega * (self.beta - entropy) + \
                (param_eta + param_omega) * 1 / batchsize * \
                tf.reduce_sum(tf.reduce_logsumexp((param_eta * log_action_prob + F_w) / (param_omega + param_eta), axis=1))
        #        tf.reduce_sum(tf.log(tf.reduce_sum(tf.exp((param_eta * log_action_prob + F_w) / (param_omega + param_eta)), axis=1)))

        # Symbolic dual gradient
        dual_grad = tf.gradients(xs=[param_eta, param_omega], ys=dual)

        # Eval functions.
        f_dual = U.function(
            inputs=[F_w, log_action_prob] + [param_eta, param_omega, batchsize, entropy],
            outputs=dual
        )

        f_dual_grad = U.function(
            inputs=[F_w, log_action_prob] + [param_eta, param_omega, batchsize, entropy],
            outputs=dual_grad
        )

        self.opt_info = dict(
            f_dual=f_dual,
            f_dual_grad=f_dual_grad,
        )
