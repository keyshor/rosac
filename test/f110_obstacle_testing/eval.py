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


def eval_end_to_end(automaton, max_steps_in_mode, use_mode_pred, save_path):

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
        (m_name, pred_mode): []
        for m_name in automaton.modes
        for pred_mode in automaton.modes
    }

    for _ in range(20):
        observation = env.reset()
        mode = env.mode.name
        e = 0
        e_in_mode = 0
        done = False
        while not done and e_in_mode < max_steps_in_mode:
            e += 1
            e_in_mode += 1
            new_mode = mode_predictor.get_mode(observation) \
                if use_mode_pred else env.mode.name
            if mode != new_mode:
                e_in_mode = 0
                mode = new_mode
            action = controllers[mode].get_action(observation)
            state_history[env.mode.name, mode].append(env.state)
            observation, reward, done, info = env.step(action)
        if e >= max_steps_in_mode:
            print(f'stuck in mode {mode} after {e} steps')
        elif env.mode.is_safe(env.state):
            print(f'terminated normally after {e} steps')
        else:
            print(f'crash after {e} steps in mode {mode}')

    for (m_name, m) in automaton.modes.items():
        fig, ax = plt.subplots(figsize=(12, 10))
        m.plot_halls(ax=ax, st=m.reset())
        colors = ['r', 'g', 'b', 'm', 'c', 'y']
        for (pred_mode, c) in zip(list(automaton.modes), colors):
            x_hist = [s.x for s in state_history[m_name, pred_mode]]
            y_hist = [s.y for s in state_history[m_name, pred_mode]]
            ax.scatter(x_hist, y_hist, s=1, c=c, label=pred_mode)
        ax.legend(markerscale=10)
        ax.set_title(m_name)
        ax.set_aspect('equal')
        fig.savefig(f'trajectories_{m.name}.png')

def eval_single(automaton, name, save_path):
    mode = automaton.modes[name]
    env = GymEnvWrapper(mode, automaton.transitions[name])
    controller = BaselineCtrlWrapper.load(
        os.path.join(save_path, name, 'best_model.zip'),
        algo_name='td3',
        policy=F110Policy,
        env=env,
    )
    state_history = []
    nonterm = 0
    normal = 0
    crash = 0
    for _ in range(100):
        observation = env.reset()
        e = 0
        done = False
        while not done:
            if e > 100:
                break
            e += 1
            action = controller.get_action(observation)
            state_history.append(env.state)
            observation, reward, done, info = env.step(action)
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
    mode.plot_halls(ax=ax, st=state_history[0])
    x_hist = [s.x for s in state_history]
    y_hist = [s.y for s in state_history]
    ax.scatter(x_hist, y_hist, s=1)
    ax.set_title(name)
    ax.set_aspect('equal')
    fig.savefig(f'trajectories_{name}.png')
    print(f'{name}: nonterm = {nonterm}, normal = {normal}, crash = {crash}')

def plot_lidar(automaton, name):
    controller = BaselineCtrlWrapper.load(f'{name}.td3', algo_name='td3', policy=F110Policy)
    mode = automaton.modes[name]
    st = mode.reset()
    mode.render(st)

if __name__ == '__main__':
    automaton = make_f110_model(straight_lengths=[10])
    ap = argparse.ArgumentParser(description='evaluate controllers and mode predictor')
    ap.add_argument('--path', type=pathlib.Path, default='.',
                    help='directory in which models are stored')
    ap.add_argument('--all', action='store_true',
                    help='use this flag to evaluate instead of specifying a list')
    ap.add_argument('--no-mode-pred', action='store_true',
                    help='disables mode predictor in end-to-end evaluation')
    ap.add_argument('--no-e2e', action='store_true',
                    help='disables end-to-end evaluation')
    ap.add_argument('--mode-length', type=int, default=100,
                    help='maximum number of time-steps in each mode')
    ap.add_argument('modes', type=str, nargs='*',
                    help='modes for which controllers will be evaluated')
    args = ap.parse_args()

    mode_list = list(automaton.modes) if args.all else args.modes
    for name in mode_list:
        eval_single(automaton, name, args.path)
    if not args.no_e2e:
        eval_end_to_end(automaton, args.mode_length, not args.no_mode_pred, args.path)
