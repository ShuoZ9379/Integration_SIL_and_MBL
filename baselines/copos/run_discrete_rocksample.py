#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI

from baselines.common import set_global_seeds
from baselines import logger
from baselines.common.cmd_util import arg_parser
from baselines.copos.compatible_mlp_policy_discrete import CompatibleMlpPolicyDiscrete
from baselines.copos import copos_mpi
from baselines.env.history_env import HistoryEnv

import tensorflow as tf
import numpy as np

import gym
import gym_pomdp


def train_copos(env_id, num_timesteps, seed):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()


    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    def policy_fn(name, ob_space, ac_space):
        return CompatibleMlpPolicyDiscrete(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=32, num_hid_layers=2)

    set_global_seeds(workerseed)
    # types: field_vision_pos, standard, fully_observable, ...
    env = HistoryEnv(env_id, hist_len=1, history_type='fully_observable', kwargs={'board_size':5, 'num_rocks':7})
    #env = HistoryEnv(env_id, hist_len=15)
    print("ob_space: "+str(env.observation_space))
    print("ac_space: "+str(env.action_space))    
    env.seed(workerseed)

    #timesteps_per_batch=1024
    #timesteps_per_batch=2048
    timesteps_per_batch=5000
    beta = -1
    if beta < 0:
        nr_episodes = num_timesteps // timesteps_per_batch
        # Automatically compute beta based on initial entropy and number of iterations
        tmp_pi = policy_fn("tmp_pi", env.observation_space, env.action_space)

        sess.run(tf.global_variables_initializer())

        tmp_ob = np.zeros((1,) + env.observation_space.shape)
        entropy = sess.run(tmp_pi.pd.entropy(), feed_dict={tmp_pi.ob: tmp_ob})
        #beta = 2 * entropy / nr_episodes
        beta = entropy / nr_episodes
        print("Initial entropy: " + str(entropy) + ", episodes: " + str(nr_episodes))
        print("Automatically set beta: " + str(beta))
    #beta = 0.01
    beta = 0.005
    epsilon = beta

    copos_mpi.learn(env, policy_fn, timesteps_per_batch=timesteps_per_batch, epsilon=epsilon, beta=beta, cg_iters=10, cg_damping=0.1,
                    max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3, TRPO=False)
    env.close()

def argparser():
    """
    Create an argparse.ArgumentParser.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Rock-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    #parser.add_argument('--log_dir',type=str, default='../logs')
    return parser

def main():
    args = argparser().parse_args()
    #logger.configure(dir=args.log_dir)
    train_copos(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()

