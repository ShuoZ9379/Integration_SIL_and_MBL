#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
import argparse

from mpi4py import MPI

from baselines.common import set_global_seeds
from baselines import logger
from baselines.common.cmd_util import arg_parser, args_dict_to_csv
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.n_copos.compatible_mlp_policy import CompatibleMlpPolicy
from baselines.n_copos import copos_mpi

import tensorflow as tf
import numpy as np

import roboschool
import gym


def train_copos(env_id, compatible_policy, num_timesteps, timesteps_per_batch, seed, filepath, visualize, n_policy, retrace, trpo,
                entropy_bonus, epsilon, beta):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()


    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure(dir=filepath)
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)
    workerseed = seed # + 10000 * MPI.COMM_WORLD.Get_rank()
    if compatible_policy:
        def policy_fn(name, ob_space, ac_space):
            return CompatibleMlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=32, num_hid_layers=2)
    else:
        assert(trpo)
        def policy_fn(name, ob_space, ac_space):
            return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=32, num_hid_layers=2)

    set_global_seeds(workerseed)
    env = gym.make(env_id)
    env.seed(workerseed)

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
        copos_mpi.visualize(env, policy_fn, timesteps_per_batch=timesteps_per_batch, epsilon=epsilon, beta=beta, cg_iters=10, cg_damping=0.1,
                            max_timesteps=num_timesteps, gamma=0.99, lam=0.98, entcoeff=entropy_bonus, vf_iters=5,
                            vf_stepsize=1e-3, TRPO=trpo, n_policy=n_policy,
                            policy_type=1, filepath=filepath, session=sess, retrace=retrace)
        env.close()
    else:
        # Train policy and save it
        copos_mpi.learn(env, policy_fn, timesteps_per_batch=timesteps_per_batch, epsilon=epsilon, beta=beta, cg_iters=10, cg_damping=0.1,
                        max_timesteps=num_timesteps, gamma=0.99, lam=0.98, entcoeff=entropy_bonus, vf_iters=5,
                        vf_stepsize=1e-3, TRPO=trpo, n_policy=n_policy,
                        policy_type=1, filepath=filepath, session=sess, retrace=retrace)
        env.close()
        saver = tf.train.Saver()
        saver.save(sess, filepath + "_final")


def argparser():
    """
    Create an argparse.ArgumentParser.
    """

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        
    parser = arg_parser()
    parser.add_argument('--env-id', help='environment ID', type=str, default='RoboschoolReacher-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=int(0))
    parser.add_argument('--num-timesteps', type=int, default=int(0.5e5))
    parser.add_argument('--timesteps_per_episode', type=int, default=int(10000))
    parser.add_argument('--n_policy', help='Number of policies to execute', type=int, default=int(1))
    parser.add_argument('--filepath', type=str, default='/tmp/')
    parser.add_argument('--visualize', help='Load and visualize experiment?', type=str2bool, default=False)
    parser.add_argument('--retrace', help='Use retrace?', type=str2bool, default=False)
    parser.add_argument('--trpo', help='Use TRPO instead of COPOS?', type=str2bool, default=False)
    parser.add_argument('--entropy_bonus', help='Entropy bonus factor', type=float, default=float(0.0))
    parser.add_argument('--epsilon', help='Epsilon', type=float, default=float(0.01))
    parser.add_argument('--beta', help='Beta', type=float, default=float(0.01))
    parser.add_argument('--compatible', help='Use compatible policy?', type=str2bool, default=True)

    return parser


def main():
    args = argparser().parse_args()
    if not args.filepath:
        args.filepath = "/tmp/" + args.env + "_" + str(args.seed)

    # Save arguments to info.csv
    dict_args = vars(args).copy()
    dict_args['algo'] = 'TRPO' if args.trpo else 'COPOS'
    args_dict_to_csv(dict_args['filepath'] + "/info.csv", dict_args)

    train_copos(args.env_id, compatible_policy=args.compatible,
                num_timesteps=args.num_timesteps, timesteps_per_batch=args.timesteps_per_episode,
                seed=args.seed, filepath=args.filepath,
                visualize=args.visualize, n_policy=args.n_policy, retrace=args.retrace, trpo=args.trpo,
                entropy_bonus=args.entropy_bonus, epsilon=args.epsilon, beta=args.beta)


if __name__ == '__main__':
    main()

