#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI

from baselines.common import set_global_seeds
from baselines import logger
from baselines.common.cmd_util import arg_parser
from baselines.copos.compatible_mlp_policy import CompatibleMlpPolicy
from baselines.copos import copos_mpi
from baselines.env.envsetting import new_lunar_lander_pomdp_env

import tensorflow as tf
import numpy as np

from baselines.common.cmd_util import make_control_env
import os
import datetime

def train_copos(env_id, num_timesteps, seed, hist_len, block_high, nsteps, hid_size, give_state):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    def policy_fn(name, ob_space, ac_space):
        return CompatibleMlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=hid_size, num_hid_layers=2)

    set_global_seeds(workerseed)

    env = make_control_env(env_id, workerseed, hist_len=hist_len,
                           block_high=block_high, not_guided=True, give_state=True)
    env.seed(workerseed)

    timesteps_per_batch=nsteps

    ###TODO: The following several lines are used for evaluation
    # pi = policy_fn('pi', env.observation_space, env.action_space)
    # sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
    # saver.restore(sess, '/work/scratch/rz97hoku/ReinforcementLearning/tmp/hist4/copos-ratio/copos-ratio-1-11-05-20-11/checkpoints/00976.ckpt')
    # for m in range(100):
    #     ob = env.reset()
    #     ep_rwd = []
    #     while True:
    #         ac, _ = pi.act(stochastic=False, ob=ob)
    #         ob, rew, new, _ = env.step(ac)
    #         ep_rwd.append(rew)
    #         if new:
    #             break
    #     logger.record_tabular("Reward", np.sum(ep_rwd))
    #     logger.record_tabular("Episode", m)
    #     logger.dump_tabular()


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

    #copos_mpi.learn(env, policy_fn, timesteps_per_batch=timesteps_per_batch, epsilon=0.01,
    #                beta=beta, cg_iters=10, cg_damping=0.1, sess=sess,
    #                max_timesteps=num_timesteps, gamma=0.99,
    #                lam=0.98, vf_iters=vf_iters, vf_stepsize=1e-3, trial=trial, method=method)
    copos_mpi.learn(env, policy_fn, timesteps_per_batch=timesteps_per_batch, epsilon=0.01,
                    beta=beta, cg_iters=10, cg_damping=0.1, max_timesteps=num_timesteps, gamma=0.99,
                    lam=0.98, vf_iters=5, vf_stepsize=1e-3)
    env.close()


def get_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def save_args(args):
    for arg in vars(args):
        logger.log("{}:".format(arg), getattr(args, arg))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

def frac2float(v):
    num = v.split('/')
    return float(num[0])/float(num[1])

def str2list(v):
    net = v.split(',')
    return [int(n) for n in net]

def control_arg_parser():
    """
    Create an argparse.ArgumentParser for run_box2d.py.
    """
    parser = arg_parser()
    parser.add_argument('--log_dir',type=str, default='../logs')
    parser.add_argument('--env', help='environment ID', type=str, default='LunarLanderContinuousPOMDP-v0')
    # parser.add_argument('--net_size', help='Network size', default=[64,64], type=str2list)
    # parser.add_argument('--filter_size', help='Define filter size for modified CNN policy', default=[16, 2], type=str2list)
    parser.add_argument('--hist_len', help='History Length', type=int, default=8)
    # parser.add_argument('--block_high', help='Define the hight of shelter area, should be greater than 1/2',
    #                     default=5/8, type=frac2float)
    parser.add_argument('--block_high', help='Define the hight of shelter area, should be greater than 1/2',
                        default=3/4, type=frac2float)
    parser.add_argument('--nsteps', help='timesteps each iteration', type=int, default=2048)
    parser.add_argument('--hid_size', help='number of neurons for each hidden layer', type=int, default=32)
    # parser.add_argument('--batch_size', help='batch size', type=int, default=32)
    parser.add_argument('--method', help='method', type=str, default='copos-new-evaluation')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--give_state', help='0:False, 1:True', type=int, default=1)
    # parser.add_argument('--train', help='train', default=False, type=str2bool)
    # parser.add_argument('--render', help='render', default=False, type=str2bool)
    # parser.add_argument('--load_path', default=None)
    # parser.add_argument('--checkpoint', help='Use saved checkpoint?', default=False, type=str2bool)
    # parser.add_argument('--iters', help='Iterations so far(to produce videos)', default=0)
    # parser.add_argument('--use_entr', help='Use dynammic entropy regularization term?', default=False, type=str2bool)
    return parser

def main():
    args = control_arg_parser().parse_args()
    ENV_path = get_dir(os.path.join(args.log_dir, args.env))
    log_dir = os.path.join(ENV_path, args.method + "-" +
                           '{}'.format(args.seed)) + "-" + \
              datetime.datetime.now().strftime("%m-%d-%H-%M")
    logger.configure(dir=log_dir)
    save_args(args)
    # if args.env == 'LunarLanderContinuousPOMDP-v0':
    #     new_lunar_lander_pomdp_env(hist_len=args.hist_len, block_high=float(args.block_high), policy_name=args.policy_name)
    #train_copos(args.env, num_timesteps=args.num_timesteps * 1e6, seed=args.seed, trial=args.seed,
    #            hist_len=args.hist_len, block_high=float(args.block_high), nsteps=args.nsteps,
    #            method=args.method, hid_size=args.hid_size, give_state=args.give_state, vf_iters=args.epoch)
    train_copos(args.env, num_timesteps=args.num_timesteps, seed=args.seed, hist_len=args.hist_len, 
                block_high=float(args.block_high), nsteps=args.nsteps, hid_size=args.hid_size, 
                give_state=args.give_state)


if __name__ == '__main__':
    main()

