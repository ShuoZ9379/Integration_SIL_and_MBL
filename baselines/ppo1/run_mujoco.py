#!/usr/bin/env python3

from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common import tf_util as U
from baselines import logger

import tensorflow as tf

import roboschool

def train(env_id, num_timesteps, timesteps_per_actor_batch, seed, entropy_coeff, filepath):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = make_mujoco_env(env_id, seed)
    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=timesteps_per_actor_batch,
            clip_param=0.2, entcoeff=entropy_coeff,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear'
        )
    env.close()

    # Save policy etc.
    saver = tf.train.Saver()
    saver.save(sess, filepath + "_final")


def main():
    args = mujoco_arg_parser().parse_args()
    logger.configure(dir=args.filepath)
    train(args.env, num_timesteps=args.num_timesteps, timesteps_per_actor_batch=args.timesteps_per_episode,
          seed=args.seed, entropy_coeff=args.entropy_coeff, filepath=args.filepath)

if __name__ == '__main__':
    main()
