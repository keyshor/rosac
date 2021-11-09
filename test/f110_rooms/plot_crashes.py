import os
import sys
import argparse
import pathlib
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.util.wrappers import Sb3CtrlWrapper, GymEnvWrapper
from hybrid_gym.envs.f110_rooms.hybrid_env import make_f110_rooms_model
from hybrid_gym.hybrid_env import HybridEnv
from hybrid_gym.selectors import UniformSelector, MaxJumpWrapper

import matplotlib.pyplot as plt


def plot_crashes(automaton, max_steps_in_mode, num_trials, save_path):

    env = HybridEnv(
        automaton=automaton,
        selector=MaxJumpWrapper(
            wrapped_selector=UniformSelector(modes=automaton.modes.values()),
            max_jumps=100
        ),
        max_timesteps=1000000000000000000000000,
        max_timesteps_per_mode=100,
    )

    controllers = {
        name: Sb3CtrlWrapper.load(
            os.path.join(save_path, name, 'best_model.zip'),
            algo_name='td3',
            env=env,
        )
        for name in automaton.modes
    }

    state_history = []

    num_normal = 0
    num_stuck = 0
    num_crash = 0
    survival_times = []

    num_trials_format_width = len(str(num_trials - 1))
    for i in range(num_trials):
        observation = env.reset()
        num_jumps = 0
        state_history.clear()
        done = False
        while not done:
            action = controllers[env.mode.name].get_action(observation)
            state_history.append(env.state)
            observation, reward, done, info = env.step(action)
            if info['jump']:
                num_jumps += 1
                state_history.clear()
        plot_trajectory = True
        if info['stuck']:
            num_stuck += 1
            print(f'stuck in mode {env.mode.name} after {num_jumps} jumps')
        elif env.mode.is_safe(env.state):
            num_normal += 1
            plot_trajectory = False
            print(f'terminated normally after {num_jumps} jumps')
        else:
            num_crash += 1
            print(f'crash after {num_jumps} jumps in mode {env.mode.name}')
        survival_times.append(num_jumps)

        if plot_trajectory:
            fig, ax = plt.subplots()
            m = env.mode
            st = state_history[0]
            m.plot_halls(ax=ax, st=st)
            x_hist = [s.x for s in state_history]
            y_hist = [s.y for s in state_history]
            ax.scatter(x_hist, y_hist, s=1)
            ax.set_title(f'trajectory {i:0{num_trials_format_width}d}, mode {m.name}')
            ax.set_aspect('equal')
            fig.savefig(f'trajectories_{i:0{num_trials_format_width}d}.png')
            plt.close(fig)
    avg_survival = sum(survival_times) / len(survival_times)
    print(f'normal {num_normal}, stuck {num_stuck}, crash {num_crash}, survival {avg_survival}')

if __name__ == '__main__':
    automaton = make_f110_rooms_model()
    ap = argparse.ArgumentParser(description='plot crashes')
    ap.add_argument('--path', type=pathlib.Path, default='.',
                    help='directory in which models are stored')
    ap.add_argument('--mode-length', type=int, default=100,
                    help='maximum number of time-steps in each mode')
    ap.add_argument('--num-trials', type=int, default=20,
                    help='number of trials')
    args = ap.parse_args()

    plot_crashes(automaton, args.mode_length, args.num_trials, args.path)
