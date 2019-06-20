import sys
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np
import pickle
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module

from mbl.util.vec_normalize import VecNormalize
from cmd_util import parse_unknown_args, make_vec_env, make_env, common_arg_parser

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
    env_type, env_id = get_env_type(args.env)
    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args, normalize_ob=args.normalize_env_obs, normalize_ret=args.normalize_env_ret, is_eval=False, num_env=1)
    eval_env = build_env(args, normalize_ob=args.normalize_env_obs, normalize_ret=args.normalize_env_ret, is_eval=True, num_env=1)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))
    
    iters = 0
    for model in learn(
                env=env,
                env_id=env_id,
                eval_env=eval_env,
                make_eval_env=lambda: build_env(args, normalize_ob=args.normalize_env_obs, normalize_ret=args.normalize_env_ret, is_eval=True, num_env=1),
                seed=seed,
                total_timesteps=total_timesteps,
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


def build_env(args, normalize_ob, normalize_ret, is_eval=False, num_env=1):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args.env)
    print(env_type)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(allow_soft_placement=True,
                               gpu_options=gpu_options,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)

        get_session(config=config)

        env = make_vec_env(env_id, env_type, num_env or 1, seed, reward_scale=args.reward_scale)

        if env_type == 'mujoco':
            logger.log('build_env: normalize_ob', normalize_ob)
            env = VecNormalize(env, ob=normalize_ob, ret=(not is_eval) and (normalize_ret), is_training=not is_eval)

    return env


def get_env_type(env_id):
    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type == 'atari':
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    print(submodule)
    # first try to import the alg module from baselines
    alg_module = import_module('.'.join([submodule]))

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



def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}



def main():
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args()
    extra_args = parse_cmdline_kwargs(unknown_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure()
        logger.configure(dir=extra_args['logdir'])
    else:
        logger.configure(format_strs=[])
        rank = MPI.COMM_WORLD.Get_rank()

    model, env = train(args, extra_args)
    env.close()

    #if args.save_path is not None and rank == 0:
    #    save_path = osp.expanduser(args.save_path)
    #    model.save(save_path)

    if args.play:
        logger.log("Running trained model")
        env = build_env(args)
        obs = env.reset()
        def initialize_placeholders(nlstm=128,**kwargs):
            return np.zeros((args.num_env or 1, 2*nlstm)), np.zeros((1))
        state, dones = initialize_placeholders(**extra_args)
        while True:
            actions, _, state, _ = model.step(obs,S=state, M=dones)
            obs, _, done, _ = env.step(actions)
            env.render()
            done = done.any() if isinstance(done, np.ndarray) else done

            if done:
                obs = env.reset()

        env.close()

if __name__ == '__main__':
    main()
