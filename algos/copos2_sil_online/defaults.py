from baselines.common.models import mlp, cnn_small


def atari():
    return dict(
        network = cnn_small(),
        timesteps_per_batch=512,
        epsilon=0.001,
        cg_iters=10,
        cg_damping=1e-3,
        gamma=0.98,
        lam=1.0,
        vf_iters=3,
        vf_stepsize=1e-4,
        entcoeff=0.00,
    )

def sparse_mujoco():
    return dict(
        network = mlp(num_hidden=32, num_layers=2),
        timesteps_per_batch=1024,
        epsilon=0.01,
        cg_iters=10,
        cg_damping=0.1,
        gamma=0.99,
        lam=0.98,
        vf_iters=5,
        vf_stepsize=1e-3,
        entcoeff=1.0,
        normalize_observations=True,
    )

def mujoco():
    return dict(
        network = mlp(num_hidden=32, num_layers=2),
        timesteps_per_batch=2048,
        epsilon=0.01,
        cg_iters=10,
        cg_damping=0.1,
        gamma=0.99,
        lam=0.98,
        vf_iters=5,
        vf_stepsize=1e-3,
        normalize_observations=True,
        entcoeff=0.0,
   #     lr=3e-4,
        lr=lambda f: 3e-4 * f,
    )


def gym_locomotion_envs():
    return dict(
        network = mlp(num_hidden=32, num_layers=2),
        timesteps_per_batch=1024,
        epsilon=0.01,
        cg_iters=10,
        cg_damping=0.1,
        gamma=0.99,
        lam=0.98,
        vf_iters=5,
        vf_stepsize=1e-3,
        normalize_observations=True,
        entcoeff=0.0,
    )

def robotics():
    return dict(
        network = mlp(num_hidden=32, num_layers=2),
        timesteps_per_batch=1024,
        epsilon=0.01,
        cg_iters=10,
        cg_damping=0.1,
        gamma=0.99,
        lam=0.98,
        vf_iters=5,
        vf_stepsize=1e-3,
        normalize_observations=True,
        entcoeff=0.0,
    )

def classic_control():
    return dict(
        network = mlp(num_hidden=32, num_layers=2),
        timesteps_per_batch=1024,
        #epsilon=0.01,
        #cg_iters=10,
        #cg_damping=0.1,
        #gamma=0.99,
        #lam=0.98,
        #vf_iters=5,
        #vf_stepsize=1e-3,
        normalize_observations=True,
    )
