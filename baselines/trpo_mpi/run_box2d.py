#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI

from baselines.common import set_global_seeds
from baselines import logger
from baselines.common.cmd_util import arg_parser
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi
from baselines.env.envsetting import new_lunar_lander_pomdp_env

import tensorflow as tf
import numpy as np

from baselines.common.cmd_util import make_control_env
import os
import datetime

def train_trpo(env_id, num_timesteps, seed, hist_len, block_high, nsteps, hid_size, give_state):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=hid_size, num_hid_layers=2)

    set_global_seeds(workerseed)

    env = make_control_env(env_id, workerseed, hist_len=hist_len,
                           block_high=block_high, not_guided=True, give_state=False)
    env.seed(workerseed)

    timesteps_per_batch=nsteps

    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=timesteps_per_batch, max_kl=0.01, cg_iters=10, cg_damping=0.1,
                    max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
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
    parser.add_argument('--method', help='method', type=str, default='trpo-new-evaluation')
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

    train_trpo(args.env, num_timesteps=args.num_timesteps, seed=args.seed, hist_len=args.hist_len, 
                block_high=float(args.block_high), nsteps=args.nsteps, hid_size=args.hid_size, 
                give_state=args.give_state)


if __name__ == '__main__':
    main()

