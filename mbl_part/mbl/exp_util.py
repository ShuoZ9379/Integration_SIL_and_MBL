import numpy as np
import time

class Policy(object):
    def __init__(self, step, reset):
        self.step = step
        self.reset = reset

def eval_policy(env, pi, num_episodes, vis_eval, measure_time=False, measure_rew=False):
    total_rewards = []
    total_times = [] if measure_time else None
    total_sim_rewards = []
    for episode in range(num_episodes):               
        #env.venv.envs[0].seed(np.random.randint(low=0, high=10000)) # For vec_env
        obs = env.reset()     
        done = False
        episode_reward = 0
        if pi.reset is not None:
            pi.reset()                
        while not done:
            if measure_time: tstart = time.time()
            act, sim_rew = pi.step(obs)
            if measure_time: total_times.append(time.time() - tstart)
            if measure_rew: total_sim_rewards.append(sim_rew)
            obs, reward, done, _ = env.step(act)
            if vis_eval:
                env.render()
            episode_reward += reward
        if vis_eval:
            print('Ep rew', episode_reward)
        total_rewards.append(episode_reward)
    if measure_time: total_times = np.mean(total_times)
    if measure_rew: total_sim_rewards = np.mean(total_sim_rewards)
    return np.mean(total_rewards), total_times, total_sim_rewards


