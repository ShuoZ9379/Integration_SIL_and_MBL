#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI

from baselines.common import set_global_seeds
from baselines import logger
from baselines.common.cmd_util import arg_parser
from baselines.iper.compatible_mlp_policy import CompatibleMlpPolicy
from baselines.iper import copos_mpi
from baselines.iper.mlp_rewardFun import MlpRewardFun

import tensorflow as tf
import numpy as np

import gym

use_actions = True

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
        return CompatibleMlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=32, num_hid_layers=2)

    def rew_fn(name, ob_space, ac_space):
        return MlpRewardFun(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=32, num_hid_layers=2, use_actions=use_actions)

    set_global_seeds(workerseed)
    env = gym.make(env_id)
    env.seed(workerseed)

    timesteps_per_batch=1024
    beta = -1
    if beta < 0:
        nr_episodes = num_timesteps // timesteps_per_batch
        # Automatically compute beta based on initial entropy and number of iterations
        tmp_pi = policy_fn("tmp_pi", env.observation_space, env.action_space)

        sess.run(tf.global_variables_initializer())

        tmp_ob = np.zeros((1,) + env.observation_space.shape)
        entropy = sess.run(tmp_pi.pd.entropy(), feed_dict={tmp_pi.ob: tmp_ob})
        beta = 2 * entropy / nr_episodes
        print("Initial entropy: " + str(entropy) + ", episodes: " + str(nr_episodes))
        print("Automatically set beta: " + str(beta))

    copos_mpi.learn(env, policy_fn, rew_fn, timesteps_per_batch=timesteps_per_batch, epsilon=0.001, beta=beta, cg_iters=10, cg_damping=0.1,
                    max_timesteps=num_timesteps, gamma=0.99, lam=0.98, use_actions=use_actions)
    env.close()

def argparser():
    """
    Create an argparse.ArgumentParser.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='MountainCarContinuous-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    return parser

def main():
    args = argparser().parse_args()
    seed = np.random.randint(0,100000)
   # train_copos(args.env, num_timesteps=args.num_timesteps, seed=args.seed)
    train_copos(args.env, num_timesteps=args.num_timesteps, seed=seed)


if __name__ == '__main__':
    main()

