#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from baselines.common import set_global_seeds
from baselines import logger
from baselines.run import parse_cmdline_kwargs, build_env, configure_logger
from baselines.common.cmd_util import arg_parser
from baselines.copos.compatible_mlp_policy import CompatibleMlpPolicy
from baselines.copos import copos_mpi
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
import tensorflow as tf
import numpy as np

import gym,sys


def train_copos(args):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()
    
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])
    
    workerseed = args.seed + 10000 * MPI.COMM_WORLD.Get_rank()
    def policy_fn(name, ob_space, ac_space):
        return CompatibleMlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=32, num_hid_layers=2)

    set_global_seeds(workerseed)
    env = build_env(args,normalize_ob=True)
    #env = gym.make(args.env)
    #env.seed(workerseed)
    
    timesteps_per_batch=10000
    #timesteps_per_batch=2048
    beta = -1
    if beta < 0:
        nr_episodes = int(args.num_timesteps) // timesteps_per_batch
        # Automatically compute beta based on initial entropy and number of iterations
        tmp_pi = policy_fn("tmp_pi", env.observation_space, env.action_space)

        sess.run(tf.global_variables_initializer())

        tmp_ob = np.zeros((1,) + env.observation_space.shape)
        entropy = sess.run(tmp_pi.pd.entropy(), feed_dict={tmp_pi.ob: tmp_ob})
        beta = 2 * entropy / nr_episodes
        print("Initial entropy: " + str(entropy) + ", episodes: " + str(nr_episodes))
        print("Automatically set beta: " + str(beta))
    copos_mpi.learn(env, policy_fn, timesteps_per_batch=timesteps_per_batch, epsilon=0.01, beta=beta, cg_iters=10, cg_damping=0.1,
                    max_timesteps=int(args.num_timesteps), gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
    env.close()

def main(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)
    print(args)
    #args.env="MountainCarContinuous-v0"
    train_copos(args)

if __name__ == '__main__':
    main(sys.argv)

