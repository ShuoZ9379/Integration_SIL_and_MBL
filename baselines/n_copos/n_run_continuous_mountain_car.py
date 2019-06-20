#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI

from baselines.common import set_global_seeds
from baselines import logger
from baselines.common.cmd_util import arg_parser
from baselines.n_copos.compatible_mlp_policy import CompatibleMlpPolicy
from baselines.n_copos import copos_mpi

import tensorflow as tf
import numpy as np

import roboschool
import gym


def train_copos(env_id, num_timesteps, timesteps_per_batch, seed, filepath, visualize, n_policy, retrace):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()


    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)
    workerseed = seed # + 10000 * MPI.COMM_WORLD.Get_rank()
    def policy_fn(name, ob_space, ac_space):
        return CompatibleMlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=32, num_hid_layers=2)

    set_global_seeds(workerseed)
    env = gym.make(env_id)
    env.seed(workerseed)

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

    if visualize:
        # Load existing policy and visualize
        copos_mpi.visualize(env, policy_fn, timesteps_per_batch=timesteps_per_batch, epsilon=0.01, beta=beta, cg_iters=10, cg_damping=0.1,
                            max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3, TRPO=False, n_policy=n_policy,
                            policy_type=1, filepath=filepath, session=sess, retrace=retrace)
        env.close()
    else:
        # Train policy and save it
        copos_mpi.learn(env, policy_fn, timesteps_per_batch=timesteps_per_batch, epsilon=0.01, beta=beta, cg_iters=10, cg_damping=0.1,
                        max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3, TRPO=False, n_policy=n_policy,
                        policy_type=1, filepath=filepath, session=sess, retrace=retrace)
        env.close()
        saver = tf.train.Saver()
        saver.save(sess, filepath + "_final")


def argparser():
    """
    Create an argparse.ArgumentParser.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='MountainCarContinuous-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(0.5e5))
    parser.add_argument('--timesteps_per_episode', type=int, default=int(10000))
    parser.add_argument('--n_policy', help='Number of policies to execute', type=int, default=1)
    parser.add_argument('--filepath', type=str, default='/tmp/')
    parser.add_argument('--visualize', help='Load and visualize experiment?', type=bool, default=False)
    parser.add_argument('--retrace', help='Use retrace?', type=bool, default=False)
    parser.add_argument('--trpo', help='Use TRPO instead of COPOS?', type=bool, default=False)

    return parser


def main():
    args = argparser().parse_args()
    if not args.filepath:
        args.filepath = "/tmp/" + args.env + "_" + str(args.seed)

    train_copos(args.env, num_timesteps=args.num_timesteps, timesteps_per_batch=args.timesteps_per_episode,
                seed=args.seed, filepath=args.filepath,
                visualize=args.visualize, n_policy=args.n_policy, retrace=args.retrace)


if __name__ == '__main__':
    main()

