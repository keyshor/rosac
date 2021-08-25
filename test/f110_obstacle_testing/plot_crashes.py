import os
import sys
import argparse
import pathlib
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.util.wrappers import BaselineCtrlWrapper, GymEnvWrapper
from hybrid_gym.train.mode_pred import ScipyModePredictor
from hybrid_gym.envs.f110_obstacle_testing.hybrid_env import make_f110_model
from hybrid_gym.hybrid_env import HybridEnv
from hybrid_gym.selectors import UniformSelector, MaxJumpWrapper
from train import F110Policy

import matplotlib.pyplot as plt


automaton = make_f110_model(straight_lengths=[10])

def plot_crashes(automaton, max_steps_in_mode, use_mode_pred, num_trials, save_path):

    env = HybridEnv(
        automaton=automaton,
        selector=MaxJumpWrapper(
            wrapped_selector=UniformSelector(modes=automaton.modes.values()),
            max_jumps=100
        )
    )

    controllers = {
        name: BaselineCtrlWrapper.load(
            os.path.join(save_path, name, 'best_model.zip'),
            algo_name='td3',
            policy=F110Policy,
            env=env,
        )
        for name in automaton.modes
    }
    if use_mode_pred:
        mode_predictor = ScipyModePredictor.load(
            os.path.join(save_path, 'mode_predictor.mlp'),
            automaton.observation_space, 'mlp',
        )

    state_history: dict = {
        pred_mode: []
        for pred_mode in automaton.modes
    }

    for i in range(20):
        observation = env.reset()
        mode = mode_predictor.get_mode(observation) \
            if use_mode_pred else env.mode.name
        true_mode = env.mode.name
        e = 0
        e_in_mode = 0
        for hist in state_history.values():
            hist.clear()
        done = False
        while not done and e_in_mode < max_steps_in_mode:
            e += 1
            e_in_mode += 1
            mode = mode_predictor.get_mode(observation) \
                if use_mode_pred else env.mode.name
            new_true_mode = env.mode.name
            if true_mode != new_true_mode:
                e_in_mode = 0
                true_mode = new_true_mode
                for hist in state_history.values():
                    hist.clear()
            action = controllers[mode].get_action(observation)
            state_history[mode].append(env.state)
            observation, reward, done, info = env.step(action)
        plot_trajectory = True
        if e >= max_steps_in_mode:
            print(f'stuck in mode {mode} after {e} steps')
        elif env.mode.is_safe(env.state):
            print(f'terminated normally after {e} steps')
            plot_trajectory = False
        else:
            print(f'crash after {e} steps in mode {mode}')

        if plot_trajectory:
            fig, ax = plt.subplots()
            m = automaton.modes[mode]
            st = m.reset()
            for hist in state_history.values():
                try:
                    st = hist[0]
                    break
                except IndexError:
                    pass
            m.plot_halls(ax=ax, st=st)
            colors = ['r', 'g', 'b', 'm', 'c', 'y']
            for (pred_mode, c) in zip(list(automaton.modes), colors):
                x_hist = [s.x for s in state_history[pred_mode]]
                y_hist = [s.y for s in state_history[pred_mode]]
                ax.scatter(x_hist, y_hist, s=1, c=c, label=pred_mode)
            ax.legend(markerscale=10)
            ax.set_title(f'trajectory {i}, mode {mode}')
            ax.set_aspect('equal')
            fig.savefig(f'trajectories_{i}.png')
            plt.close(fig)

if __name__ == '__main__':
    automaton = make_f110_model(straight_lengths=[10])
    ap = argparse.ArgumentParser(description='plot crashes')
    ap.add_argument('--path', type=pathlib.Path, default='.',
                    help='directory in which models are stored')
    ap.add_argument('--no-mode-pred', action='store_true',
                    help='disables mode predictor in end-to-end evaluation')
    ap.add_argument('--mode-length', type=int, default=100,
                    help='maximum number of time-steps in each mode')
    ap.add_argument('--num-trials', type=int, default=20,
                    help='number of trials')
    args = ap.parse_args()

    plot_crashes(automaton, args.mode_length, not args.no_mode_pred, args.num_trials, args.path)
