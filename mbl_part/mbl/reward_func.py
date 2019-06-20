import gym
import numpy as np
import time
import math
import copy

def _bullet_calc_potential(body_y, body_x, walk_target_y, walk_target_x, dt):
    diff_y = (np.ones_like(body_y) * walk_target_y - body_y)[:, None]
    diff_x = (np.ones_like(body_x) * walk_target_x - body_x)[:, None]
    diff = np.concatenate([diff_y, diff_x], axis=1)
    walk_target_dist = np.linalg.norm(diff, axis=1)
    return -walk_target_dist / dt

def bullet_locomotion(s, a, sp, alive_bonus, walker_target_y, walker_target_x, dt, j_start, j_end):
    state = sp
    _alive = alive_bonus(state[:, 2], state[:, -1])   # state[2] is body height above ground, body_rpy[1] is pitch
    alive_done = (_alive < 0).astype(np.float32)
    infinite_done = (np.ones_like(alive_done) - np.isfinite(state).all(axis=1))
    done = np.concatenate([alive_done[:, None], infinite_done[:, None]], axis=1).any(axis=1).astype(np.float32)
    potential_old = _bullet_calc_potential(s[:, 1], s[:, 0], walker_target_y, walker_target_x, dt)
    potential = _bullet_calc_potential(sp[:, 1], sp[:, 0], walker_target_y, walker_target_x, dt)
    progress = potential - potential_old

    j = sp[:, j_start:j_end]
    joint_speeds = j[:, 1::2]
    electricity_cost  = -2.0 * np.abs(a * joint_speeds).mean(axis=1)  # let's assume we have DC motor with controller, and reverse current braking
    electricity_cost += -0.1 * (np.square(a).mean(axis=1))

    joints_at_limit = np.count_nonzero(np.abs(j[:, 0::2]) > 0.99, axis=1)
    joints_at_limit_cost = -0.1 * joints_at_limit
    rewards = _alive + progress + electricity_cost + joints_at_limit_cost

    return rewards, done

def bullet_ant(s, a, sp):
    def alive_bonus(z, pitch):
        mask = (z > 0.26).astype(np.float32)
        return 1.0 * (mask) + -1.0 * (np.ones_like(mask) - mask)
    return bullet_locomotion(s, a, sp, alive_bonus, 0, 1e3, 0.01, 10, 26)

def bullet_halfcheetah(s, a, sp):
    def alive_bonus(z, pitch):
        feet_contact = sp[:, 22:22+6]
        c0 = (np.abs(pitch) < 1.0)[:, None]
        c1 = np.logical_not(feet_contact[:, 1])[:, None]
        c2 = np.logical_not(feet_contact[:, 2])[:, None]
        c3 = np.logical_not(feet_contact[:, 4])[:, None]
        c4 = np.logical_not(feet_contact[:, 5])[:, None]
        mask = np.concatenate([c0, c1, c2, c3, c4], axis=1).all(axis=1)
        return (mask) * 1.0 + (np.logical_not(mask)) * -1.0
     
 
humanoid_body_mass = np.array([0.        , 8.32207894, 2.03575204, 5.85278711, 4.52555626,
            2.63249442, 1.76714587, 4.52555626, 2.63249442, 1.76714587,
            1.59405984, 1.19834313, 1.59405984, 1.19834313])

USE_SPARSE_ROBOTICS_ENV = False

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def gym_fetch_sparse(s, a, sp, ts):
    ag = s[:, -6:-3]
    dg = s[:, -3:]
    d = goal_distance(ag, dg)   
    return -(d > ts).astype(np.float32), np.zeros(s.shape[0])

def gym_fetch_dense(s, a, sp):
    ag = s[:, -6:-3]
    dg = s[:, -3:]
    d = goal_distance(ag, dg)    
    return -d, np.zeros(s.shape[0])

def gym_fetchreach(s, a, sp):
    if USE_SPARSE_ROBOTICS_ENV: 
        return gym_fetch_sparse(s, a, sp, 0.05)
    return gym_fetch_dense(s, a, sp)

def gym_fetchpush(s, a, sp):
    if USE_SPARSE_ROBOTICS_ENV: 
        return gym_fetch_sparse(s, a, sp, 0.05)
    return gym_fetch_dense(s, a, sp)

def gym_fetchslide(s, a, sp):
    if USE_SPARSE_ROBOTICS_ENV: 
        return gym_fetch_sparse(s, a, sp, 0.05)
    return gym_fetch_dense(s, a, sp)

def gym_fetchpickandplace(s, a, sp):
    if USE_SPARSE_ROBOTICS_ENV: 
        return gym_fetch_sparse(s, a, sp, 0.05)
    return gym_fetch_dense(s, a, sp)

def gym_humanoidstandup(s, a, sp):
    pos_after = sp[:, 2]

    uph_cost = (pos_after - np.zeros_like(pos_after)) / 0.003
    
    data_ctrl_size = 17
    data_ctrl = sp[:, -data_ctrl_size:]
    quad_ctrl_cost = 0.1 * np.square(data_ctrl).sum(axis=1)

    data_cfrc_ext_size = int(14 * 6)
    data_cfrc_ext = sp[:, -(data_cfrc_ext_size + data_ctrl_size):-(data_ctrl_size)]
    quad_impact_cost = .5e-6 * np.square(data_cfrc_ext).sum(axis=1)

    dummy_10 = np.ones_like(quad_impact_cost) * 10
    quad_impact_cost = np.concatenate([quad_impact_cost[:, np.newaxis], dummy_10[:, np.newaxis]], axis=1)
    quad_impact_cost = np.min(quad_impact_cost, axis=1)
   
    reward = uph_cost - quad_ctrl_cost - quad_impact_cost + np.ones_like(quad_ctrl_cost)
    
    done = np.zeros(s.shape[0])
    return reward, done

def gym_humanoid(s, a, sp):
    def mass_center(xipos):
        xipos = xipos.reshape(len(xipos), 14, 3)
        mass = np.expand_dims(humanoid_body_mass, 1)
        return (np.sum(mass * xipos, 1) / np.sum(mass))[:, 0]
    xipos_size = int(14 * 3)
    pos_before = mass_center(s[:, :xipos_size])
    pos_after = mass_center(sp[:, :xipos_size])

    cfrc_ext_size = int(14 * 6)
    ctrl_size = int(17)
    cfrc_ext = sp[:, -(cfrc_ext_size + ctrl_size):-ctrl_size]    
    ctrl = sp[:, -ctrl_size:]

    #alive_bonus = np.ones_like(len(s)) * 5.0
    alive_bonus = np.ones_like(len(s)) * 1e-3
    lin_vel_cost = 0.25 * (pos_after - pos_before) / (0.003)
    quad_ctrl_cost = 0.1 * np.square(ctrl).sum(axis=1)
    quad_impact_cost = .5e-6 * np.square(cfrc_ext).sum(axis=1)
    quad_impact_cost = np.min(np.concatenate([quad_impact_cost[:, np.newaxis], (np.ones_like(quad_impact_cost) * 10)[:, np.newaxis]], axis=1), axis=1)
    reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
   
    qpos_size = 24
    qpos = sp[:, xipos_size:xipos_size + qpos_size]
    done = np.concatenate([(qpos[:, 2] < 1.0)[:, np.newaxis], (qpos[:, 2] > 2.0)[:, np.newaxis]], axis=1)
    done = np.any(done, axis=1).astype(np.float32)
    return reward, done

def gym_reacher(s, a, sp):
    vec = s[:, -1]
    reward_dist = - np.linalg.norm(vec)
    reward_ctrl = 1e-2 * - np.square(a).sum(axis=1)
    reward = reward_dist + reward_ctrl
    done = np.zeros(len(s))
    return reward, done

def gym_swimmer(s, a, sp):
    ctrl_cost_coeff = 0.0001
    xposbefore = s[:, 0]
    xposafter = sp[:, 0]
    reward_fwd = (xposafter - xposbefore) / (0.01 * 4)
    reward_ctrl = - ctrl_cost_coeff * np.square(a).sum(axis=1)
    reward = reward_fwd + 0.1 * reward_ctrl
    return reward, np.zeros(s.shape[0])

def gym_hopper(s, a, sp):
    posbefore = s[:, 0]
    posafter = sp[:, 0]
    height = sp[:, 1]
    ang = sp[:, 2]
    alive_bonus = np.ones(s.shape[0])
    reward = (posafter - posbefore) / (0.002 * 4)
    #reward += alive_bonus
    reward -= 1e-3 * np.square(a).sum(axis=1)
    
    notdone = np.isfinite(sp).all(axis=1).astype(np.float32) * (np.abs(sp[:, 2:]) < 100).all(axis=1).astype(np.float32) * (height > .7).astype(np.float32) * (np.abs(ang) < .2).astype(np.float32)
    done = np.ones_like(notdone) - notdone
    return reward, done

def gym_pusher(s, a, sp):
    vec_1 = sp[:, -6:-3] - sp[:, -9:-6]
    vec_2 = sp[:, -6:-3] - sp[:, -3:]

    reward_near = - np.linalg.norm(vec_1, axis=1)
    reward_dist = - np.linalg.norm(vec_2, axis=1)
    reward_ctrl = - np.square(a).sum(axis=1)
    reward = 1.25 * reward_dist + 1e-3 * 0.1 * reward_ctrl + 1e-3 * 0.5 * reward_near
    done = np.zeros(s.shape[0])
    return reward, done

def gym_walker2d(s, a, sp):
    posbefore = s[:, 0]
    posafter = sp[:, 0]
    height = sp[:, 1]
    ang = sp[:, 2]

    alive_bonus = np.ones(s.shape[0])
    reward = ((posafter - posbefore) / (0.002 * 4))
    #reward += alive_bonus
    reward -= 1e-3 * np.square(a).sum(axis=1)
    notdone = (height > 0.8).astype(np.float32) * (height < 2.0).astype(np.float32) * (ang > -1.0).astype(np.float32) * (ang < 1.0).astype(np.float32)
    done = np.ones_like(notdone) - notdone
    return reward, done    

def gym_ant(s, a, sp):
    xdiff = sp[:, -1] - s[:, -1]

    forward_reward = (xdiff) / (0.01 * 5)    

    ctrl_cost = .5 * np.square(a).sum(axis=1)

    contact_cost = np.zeros_like(ctrl_cost)

    survive_reward = np.ones(sp.shape[0]) 

    #reward = forward_reward - ctrl_cost - contact_cost + survive_reward
    #reward = forward_reward
    reward = forward_reward - ctrl_cost * 0.01 + survive_reward * 0.001

    notdone = np.isfinite(sp).all(axis=1).astype(np.float32) *  (sp[:, 2] >= 0.3).astype(np.float32) * (sp[:, 2] <= 1.0).astype(np.float32)
    done = np.ones_like(notdone) - notdone
    #done = np.zeros(s.shape[0])
    
    return reward, done

def gym_invertedpendulum(s, a, sp):
    ob = sp
    reward = np.ones(ob.shape[0])
    notdone = np.isfinite(ob).all(axis=1).astype(np.float32) * (np.abs(ob[:, 1]) <= .2).astype(np.float32)
    done = np.ones_like(notdone) - notdone
    return reward, done

def gym_sparsehalfcheetah(s, a, sp):
    xposbefore = s[:, 0]
    xposafter = sp[:, 0]
    reward_ctrl = - 0.1 * np.square(a).sum(axis=1)
    reward_run = (xposafter)
    
    mask = (xposafter >= 5.0).astype(np.float32)
    reward = mask * (reward_run)

    done = False
 
    return reward, np.zeros(s.shape[0])


def gym_halfcheetah(s, a, sp):
    xposbefore = s[:, 0]
    xposafter = sp[:, 0]
    reward_ctrl = 1e-4 * -0.1 * np.square(a).sum(axis=1)
    reward_run = (xposafter - xposbefore) / (0.01 * 5)
    reward = reward_ctrl + reward_run
    done = False

    return reward, np.zeros(s.shape[0])

def gym_pendulum(s, a, sp):
    def angle_normalize(x):
        return (((x + np.ones_like(x) * np.pi) % (2 * np.pi)) - (np.ones_like(x) * np.pi))

    max_speed = 8
    max_torque = 2.

    #th, thdot = sp # th := theta
    theta_cos = sp[:, 0]
    theta_sin = sp[:, 1]
    theta_dot = sp[:, 2]

    theta_cos = np.clip(theta_cos, -1.0, 1.0)
    th = np.arccos(theta_cos)
    thdot = theta_dot

    g = 10.
    m = 1.
    l = 1.
    dt = .05
    u = np.clip(a, -max_torque, max_torque)[0]
    costs = angle_normalize(th)**2 + .1 * thdot**2 + .001 * (u**2)
    
    return -costs, np.zeros(s.shape[0])

def gym_continuous_mountaincar(s, a, sp):
    position, velocity = s[:, 0], s[:, 1]
    velocity += ((a - np.ones_like(a)) * 0.001).squeeze(axis=1)
    velocity += np.cos(3 * position) * (-0.0025)

    velocity = np.clip(velocity, -0.07, 0.07)
    position += velocity
    position = np.clip(position, -1.2, 0.6)

    done = (position >= 0.5).astype(np.float32)
    reward = np.ones(s.shape[0]) * -1.0

    return reward, done

def gym_mjcartpole(s, a, sp):
    def _get_ee_pos(x):
        x0, theta = x[:, 0], x[:, 1]
        a = x0 - 0.6 * np.sin(theta)
        b = -0.6 * np.cos(theta)
        return np.stack([a, b], axis=1)

    cost_lscale = 0.6
    ee_pos = _get_ee_pos(sp) - np.array([0.0, 0.6])
    reward_obs = np.exp(-np.sum(np.square(ee_pos), axis=1) / (cost_lscale ** 2))
    reward_ctrl = -0.01 * np.sum(np.square(a), axis=1)
    reward = reward_obs + reward_ctrl 
    done = np.zeros(s.shape[0])
    return reward, done

def gym_cartpole(s, a, sp):
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = (masspole + masscart)
    length = 0.5 # actually half the pole's length
    polemass_length = (masspole * length)
    force_mag = 10.0
    tau = 0.02  # seconds between state updates

    # Angle at which to fail the episode
    theta_threshold_radians = 12 * 2 * math.pi / 360
    x_threshold = 2.4
    
    state = sp
    #x, _, theta, _ = state
    x = state[:, 0]
    theta = state[:, 2]
    conds = np.concatenate([ 
            (x < -x_threshold).astype(np.float32)[:, np.newaxis],
            (x > x_threshold).astype(np.float32)[:, np.newaxis],
            (theta < -theta_threshold_radians)[:, np.newaxis],
            (theta > theta_threshold_radians)[:, np.newaxis]], axis=1)
    dones = np.any(conds, axis=1).astype(np.float32)
    rewards = np.ones_like(dones) - dones
    
    return rewards, dones

def get_reward_done_func(env_id):
    dicts = {
            # Classic control
            'CartPole-v0': gym_cartpole,
            'Pendulum-v0': gym_pendulum,
            'MountainCarContinuous-v0': gym_continuous_mountaincar,
            # MuJoCo
            'Humanoid-v2': gym_humanoid,
            'HumanoidStandup-v2': gym_humanoidstandup,
            'HalfCheetah-v2': gym_halfcheetah,
            'SparseHalfCheetah-v2': gym_sparsehalfcheetah,
            'Ant-v2': gym_ant,
            'Walker2d-v2': gym_walker2d,
            'Hopper-v2': gym_hopper,
            'Swimmer-v2': gym_swimmer,
            'Pusher-v2': gym_pusher,
            'Reacher-v2': gym_reacher,
            'InvertedPendulum-v2': gym_invertedpendulum,
            'MjCartPole-v2': gym_mjcartpole,
            # Robotics
            'FetchReach-v1': gym_fetchreach,
            'FetchPush-v1': gym_fetchpush,
            'FetchSlide-v1': gym_fetchslide,
            'FetchPickAndPlace-v1': gym_fetchpickandplace,
            # Bullet
            'AntBulletEnv-v0': bullet_ant,
            }
    return dicts[env_id]


