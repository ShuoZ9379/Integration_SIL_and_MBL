import multiprocessing
import os.path as osp
import gym,sys
from collections import defaultdict
import tensorflow as tf
import numpy as np
import pickle
from baselines.common.vec_env import VecFrameStack,VecEnv, VecNormalize
from baselines.run import parse_cmdline_kwargs, build_env, configure_logger, get_default_network, get_env_type
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module
from baselines.common import set_global_seeds
import baselines.common.tf_util as U
from baselines.common.policies import build_policy
from baselines.common.input import observation_placeholder
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args,make_vec_env, make_env
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    if env.id.find('Sparse') > -1:
        _game_envs['sparse_{}'.format(env_type)].add(env.id)
    else:
        _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def train(args, extra_args):
    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))
    total_timesteps = int(args.num_timesteps)
    seed = args.seed
    set_global_seeds(seed)
    
    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args,normalize_ob=True)
    eval_env = build_env(args,normalize_ob=False, is_eval=True)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)
    beta = -1
    if beta < 0:
        #print(alg_kwargs)
        nr_episodes = total_timesteps // alg_kwargs['timesteps_per_batch']
        # Automatically compute beta based on initial entropy and number of iterations
        policy = build_policy(env, alg_kwargs['network'], value_network='copy', normalize_observations=alg_kwargs['normalize_observations'], copos=True)
        ob = observation_placeholder(env.observation_space)
        
        sess = U.single_threaded_session()
        sess.__enter__()
        with tf.variable_scope("tmp_pi"):
            tmp_pi = policy(observ_placeholder=ob)
        sess.run(tf.global_variables_initializer())
        
        tmp_ob = np.zeros((1,) + env.observation_space.shape)
        entropy = sess.run(tmp_pi.pd.entropy(), feed_dict={tmp_pi.X: tmp_ob})
        #beta = 2 * entropy / nr_episodes
        beta = 1e-6
        print("Initial entropy: " + str(entropy) + ", episodes: " + str(nr_episodes))
        print("Constantly set beta: " + str(beta))

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))
    iters = 0
    for model in learn(
                env=env,
                env_id=env_id,
                eval_env=eval_env,
                make_eval_env=lambda: build_env(args, normalize_ob=False, is_eval=True),
                seed=seed,
                beta=beta,
                total_timesteps=total_timesteps,
                sil_update=args.sil_update,
                sil_loss=args.sil_loss, 
                **alg_kwargs
            ):
        if args.store_ckpt:
            save_path = osp.join(logger.get_dir(), 'model-{}'.format(iters))
            model.save(save_path)       
            if isinstance(env, VecNormalize):
                rms_path = osp.join(logger.get_dir(), 'rms-{}'.format(iters))
                with open(rms_path, 'wb') as f:
                    rms = (env.ob_rms, env.ret_rms)
                    pickle.dump(rms, f)
            logger.log('Save {} model'.format(iters+1))
        iters += 1

    return model, env

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    print(submodule)
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join([submodule]))
        
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module

def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs

def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args()
    extra_args = parse_cmdline_kwargs(unknown_args)
    print(args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        #rank = 0
        #logger.configure()
        #logger.configure(dir=extra_args['logdir'])
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])

    model, env = train(args, extra_args)

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    if args.play:
        logger.log("Running trained model")
        obs = env.reset()

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        episode_rew = 0
        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs,S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            obs, rew, done, _ = env.step(actions)
            episode_rew += rew[0] if isinstance(env, VecEnv) else rew
            env.render()
            done = done.any() if isinstance(done, np.ndarray) else done
            if done:
                print('episode_rew={}'.format(episode_rew))
                episode_rew = 0
                obs = env.reset()
    env.close()
    return model

if __name__ == '__main__':
    main(sys.argv)
