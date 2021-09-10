import os
import sys
import argparse
import pathlib
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.util.wrappers import Sb3CtrlWrapper, GymEnvWrapper
from hybrid_gym.train.mode_pred import ScipyModePredictor
from hybrid_gym.envs.f110.hybrid_env import make_f110_model
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
        name: Sb3CtrlWrapper.load(
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

    num_normal = 0
    num_stuck = 0
    num_crash = 0

    num_trials_format_width = len(str(num_trials - 1))
    for i in range(num_trials):
        observation = env.reset()
        mode = mode_predictor.get_mode(observation) \
            if use_mode_pred else env.mode.name
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
            action = controllers[mode].get_action(observation)
            state_history[mode].append(env.state)
            observation, reward, done, info = env.step(action)
            if info['jump']:
                e_in_mode = 0
                for hist in state_history.values():
                    hist.clear()
        plot_trajectory = True
        if e_in_mode >= max_steps_in_mode:
            num_stuck += 1
            print(f'stuck in mode {env.mode.name} after {e} steps')
        elif env.mode.is_safe(env.state):
            num_normal += 1
            plot_trajectory = False
            print(f'terminated normally after {e} steps')
        else:
            num_crash += 1
            print(f'crash after {e} steps in mode {env.mode.name}')

        if plot_trajectory:
            fig, ax = plt.subplots()
            m = env.mode
            st = m.reset()
            for hist in state_history.values():
                try:
                    st = hist[-1]
                    break
                except IndexError:
                    pass
            try:
                m.plotHalls(ax=ax)
            except AttributeError:
                try:
                    m.plot_halls(ax=ax, st=st)
                except AttributeError:
                    print(f'mode = {mode}, true_mode = {true_mode}, type(st) = {type(st)}')
            for pred_mode in automaton.modes:
                try:
                    x_hist = [s.car_global_x for s in state_history[pred_mode]]
                    y_hist = [s.car_global_y for s in state_history[pred_mode]]
                except AttributeError:
                    x_hist = [s.x for s in state_history[pred_mode]]
                    y_hist = [s.y for s in state_history[pred_mode]]
                ax.scatter(x_hist, y_hist, s=1, label=pred_mode)
            ax.legend(markerscale=10)
            ax.set_title(f'trajectory {i:0{num_trials_format_width}d}, mode {m.name}')
            ax.set_aspect('equal')
            fig.savefig(f'trajectories_{i:0{num_trials_format_width}d}.png')
            plt.close(fig)
    print(f'normal {num_normal}, stuck {num_stuck}, crash {num_crash}')

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
