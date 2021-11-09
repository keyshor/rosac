import os
import sys
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

import numpy as np
import matplotlib.pyplot as plt

from hybrid_gym.util.wrappers import BaselineCtrlWrapper, GymEnvWrapper
from hybrid_gym.envs.f110_rooms.hybrid_env import make_f110_rooms_model
from hybrid_gym.envs.f110.obstacle_mode import State
from hybrid_gym.train.single_mode import train_stable, make_sb_model

def test_safety(mode, x_low, x_high, y_low, y_high, num_samples, rng):
    safe_x = []
    safe_y = []
    unsafe_x = []
    unsafe_y = []
    goal_x = []
    goal_y = []
    reward_x = []
    reward_y = []
    base_st = mode.reset()
    for i in range(num_samples):
        st = State(
            x = rng.uniform(x_low, x_high),
            y = rng.uniform(y_low, y_high),
            V = 0,
            theta = rng.uniform(np.radians(-180), np.radians(180)),
            obstacle_x = base_st.obstacle_x,
            obstacle_y = base_st.obstacle_y,
            lines = base_st.lines,
        )
        if mode.is_safe(st):
            safe_x.append(st.x)
            safe_y.append(st.y)
        else:
            unsafe_x.append(st.x)
            unsafe_y.append(st.y)
        if mode.goal_region.contains(st):
            goal_x.append(st.x)
            goal_y.append(st.y)
        if mode.center_reward_region.contains(st):
            reward_x.append(st.x)
            reward_y.append(st.y)
    fig, ax = plt.subplots()
    ax.plot(safe_x, safe_y, color='g', marker='.', markersize=1, linestyle='None')
    ax.plot(unsafe_x, unsafe_y, color='r', marker='.', markersize=1, linestyle='None')
    mode.plot_halls(ax, base_st)
    ax.set_aspect('equal')
    fig.savefig('safety.png')
    fig, ax = plt.subplots()
    ax.plot(goal_x, goal_y, color='g', marker='.', markersize=1, linestyle='None')
    mode.plot_halls(ax, base_st)
    ax.set_aspect('equal')
    fig.savefig('goal.png')
    fig, ax = plt.subplots()
    ax.plot(reward_x, reward_y, color='g', marker='.', markersize=1, linestyle='None')
    mode.plot_halls(ax, base_st)
    mode.plot_reward_line(ax)
    ax.set_aspect('equal')
    fig.savefig('reward.png')

def test_lidar(mode, x_low, x_high, y_low, y_high, num_samples, rng):
    for i in range(num_samples):
        base_st = mode.reset()
        st = State(
            x = rng.uniform(x_low, x_high),
            y = rng.uniform(y_low, y_high),
            V = 0,
            theta = rng.uniform(np.radians(-180), np.radians(180)),
            obstacle_x = base_st.obstacle_x,
            obstacle_y = base_st.obstacle_y,
            lines = base_st.lines,
        )
        fig, ax = plt.subplots()
        ax.set_xlim(left=x_low, right=x_high)
        ax.set_ylim(bottom=y_low, top=y_high)
        mode.plot_halls(ax, st)
        mode.plot_lidar(ax, st)
        ax.set_aspect('equal')
        fig.savefig(f'lidar_{i}.png')

def train_ctrl(mode, transitions, total_timesteps):
    model = make_sb_model(
        mode,
        transitions,
        algo_name='td3',
        action_noise_scale=4.0,
        verbose=0,
        max_episode_steps=100,
    )
    train_stable(model, mode, transitions,
                 total_timesteps=total_timesteps, algo_name='td3',
                 max_episode_steps=100)

def eval_ctrl(mode, transitions, num_trials):
    controller = BaselineCtrlWrapper.load(
        os.path.join(f'{mode.name}', 'best_model.zip'),
        algo_name='td3',
    )
    env = GymEnvWrapper(mode, transitions)
    state_history = []
    nonterm = 0
    normal = 0
    crash = 0
    for _ in range(num_trials):
        observation = env.reset()
        e = 0
        done = False
        while not done:
            if e > 100:
                break
            e += 1
            delta = controller.get_action(observation)
            state_history.append(env.state)
            observation, reward, done, info = env.step(delta)
            #print(reward)
        if e > 50:
            #print('spent more than 50 steps in the mode')
            nonterm += 1
        elif env.mode.is_safe(env.state):
            #print(f'terminated normally after {e} steps')
            normal += 1
        else:
            #print(f'crash after {e} steps')
            crash += 1
    fig, ax = plt.subplots()
    mode.plot_halls(ax=ax)
    x_hist = [s.x for s in state_history]
    y_hist = [s.y for s in state_history]
    ax.scatter(x_hist, y_hist, s=1)
    ax.set_title(mode.name)
    ax.set_aspect('equal')
    fig.savefig(f'trajectories_{mode.name}.png')
    print(f'{mode.name}: nonterm = {nonterm}, normal = {normal}, crash = {crash}')

if __name__ == '__main__':
    room_width = 5
    automaton = make_f110_rooms_model(room_width=room_width)
    mode_name = sys.argv[1]
    test_x_low, test_x_high, test_y_low, test_y_high = -1.2*room_width, 1.2*room_width, -1.2*room_width, 1.2*room_width
    mode = automaton.modes[mode_name]
    transitions = automaton.transitions[mode_name]
    rng = np.random.default_rng()
    test_safety(mode, test_x_low, test_x_high, test_y_low, test_y_high, 10000, rng)
    test_lidar(mode, test_x_low, test_x_high, test_y_low, test_y_high, 10, rng)
    #train_ctrl(mode, transitions, 200000)
    #eval_ctrl(mode, transitions, 100)
