from baselines.common.models import mlp, cnn_small
def mujoco():
    return dict(
        network = mlp(num_hidden=32, num_layers=2),
        nsteps=2048,
        nminibatches=32,
        lam=0.95,
        gamma=0.99,
        noptepochs=10,
        log_interval=1,
        ent_coef=0.0,
        #lr=lambda f: 3e-4 * f,
        #lr=3e-4,
        cliprange=0.2,
        value_network='copy',
        normalize_observations=True
    )

def atari():
    return dict(
        nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=0.1,
    )

def retro():
    return atari()

def classic_control():
    return mujoco()
