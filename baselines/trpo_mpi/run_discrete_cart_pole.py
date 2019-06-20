#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI

from baselines.common import set_global_seeds
from baselines import logger
from baselines.common.cmd_util import arg_parser
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi

import tensorflow as tf
import numpy as np

import gym


def train_trpo(env_id, num_timesteps, seed):
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
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=30, num_hid_layers=2)

    set_global_seeds(workerseed)
    env = gym.make(env_id)
    env.seed(workerseed)

    timesteps_per_batch=1024

    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=timesteps_per_batch, max_kl=0.01, cg_iters=10, cg_damping=0.1,
                    max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
    env.close()

def argparser():
    """
    Create an argparse.ArgumentParser.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='CartPole-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    return parser

def main():
    args = argparser().parse_args()
    train_trpo(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()

