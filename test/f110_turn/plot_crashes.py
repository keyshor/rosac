import os
import sys
import argparse
import pathlib
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.util.wrappers import GymEnvWrapper
from hybrid_gym.envs.f110_turn.hybrid_env import make_f110_model
from hybrid_gym.rl.sac.sac import SACController
from hybrid_gym.hybrid_env import HybridEnv
from hybrid_gym.selectors import UniformSelector, MaxJumpWrapper

import matplotlib.pyplot as plt


def make_plot(automaton, max_steps_in_mode, num_trials, save_path):

    env = HybridEnv(
        automaton=automaton,
        selector=MaxJumpWrapper(
            wrapped_selector=UniformSelector(modes=automaton.modes.values()),
            max_jumps=100,
        ),
        max_timesteps_per_mode=max_steps_in_mode,
        max_timesteps=1000000,
    )

    controllers = {
        name: SACController.load(
            os.path.join(save_path, f'{name}.pkl'),
        )
        for name in automaton.modes
    }

    state_history = []

    num_normal = 0
    num_stuck = 0
    num_crash = 0

    num_trials_format_width = len(str(num_trials - 1))
    for i in range(num_trials):
        observation = env.reset()
        mode = env.mode.name
        e = 0
        e_in_mode = 0
        num_jumps = 0
        done = False
        while not done:
            e += 1
            e_in_mode += 1
            mode = env.mode.name
            action = controllers[mode].get_action(observation)
            state_history.append(env.state)
            observation, reward, done, info = env.step(action)
            if info['jump']:
                #print(f'jump {mode} -> {env.mode.name}')
                num_jumps += 1
                e_in_mode = 0
                state_history.clear()
        plot_trajectory = True
        if info['stuck']:
            num_stuck += 1
            print(f'stuck in mode {mode} after {e} steps and {num_jumps} jumps')
        elif env.mode.is_safe(env.state):
            num_normal += 1
            plot_trajectory = False
            print(f'terminated normally after {e} steps and {num_jumps} jumps')
        else:
            num_crash += 1
            print(f'crash in step {e_in_mode} of mode {mode} after {e} steps and {num_jumps} jumps')
        if plot_trajectory:
            fig, ax = plt.subplots()
            m = env.mode
            st = m.reset()
            st = state_history[-1]
            m.plot_halls(ax=ax, st=st)
            x_hist = [s.x for s in state_history]
            y_hist = [s.y for s in state_history]
            ax.scatter(x_hist, y_hist, s=1)
            ax.set_title(f'trajectory {i:0{num_trials_format_width}d}, mode {m.name}')
            ax.set_aspect('equal')
            fig.savefig(f'trajectories_{i:0{num_trials_format_width}d}.png')
            plt.close(fig)
    print(f'normal {num_normal}, stuck {num_stuck}, crash {num_crash}')



if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='evaluate controllers')
    ap.add_argument('--path', type=pathlib.Path, default='.',
                    help='directory in which models are stored')
    #ap.add_argument('--no-throttle', action='store_true',
    #                help='use this flag to disable throttle in the environment')
    ap.add_argument('--all', action='store_true',
                    help='use this flag to evaluate instead of specifying a list')
    ap.add_argument('--num-trials', type=int, default=20,
                    help='number of trials for end-to-end evaluation')
    ap.add_argument('--mode-length', type=int, default=100,
                    help='maximum number of time-steps in each mode')
    ap.add_argument('modes', type=str, nargs='*',
                    help='modes for which controllers will be evaluated')
    args = ap.parse_args()

    automaton = make_f110_model()
    make_plot(automaton, args.mode_length, args.num_trials, args.path)
