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


def eval_end_to_end(automaton, max_steps_in_mode, num_trials, save_path):

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

    state_history: dict = {
        (m_name, pred_mode): []
        for m_name in automaton.modes
        for pred_mode in automaton.modes
    }

    num_normal = 0
    num_stuck = 0
    num_crash = 0

    for _ in range(num_trials):
        observation = env.reset()
        mode = env.mode.name
        e = 0
        num_jumps = 0
        prev_mode = 'none'
        done = False
        while not done:
            e += 1
            mode = env.mode.name
            action = controllers[mode].get_action(observation)
            state_history[env.mode.name, mode].append(env.state)
            observation, reward, done, info = env.step(action)
            if info['jump']:
                #print(f'jump {mode} -> {env.mode.name}')
                prev_mode = mode
                num_jumps += 1
        if info['stuck']:
            num_stuck += 1
            print(f'stuck after jump {prev_mode} -> {mode} after {e} steps and {num_jumps} jumps')
        elif env.mode.is_safe(env.state):
            num_normal += 1
            print(f'terminated normally after {e} steps and {num_jumps} jumps')
        else:
            num_crash += 1
            print(f'crashed after jump {prev_mode} -> {mode} after {e} steps and {num_jumps} jumps')
    print(f'normal {num_normal}, stuck {num_stuck}, crash {num_crash}')

    for (m_name, m) in automaton.modes.items():
        fig, ax = plt.subplots(figsize=(12, 10))
        m.plot_halls(ax=ax, st=state_history[m_name, m_name][0])
        for pred_mode in automaton.modes:
            x_hist = [s.x for s in state_history[m_name, pred_mode]]
            y_hist = [s.y for s in state_history[m_name, pred_mode]]
            ax.scatter(x_hist, y_hist, s=1)
        ax.set_title(m_name)
        ax.set_aspect('equal')
        fig.savefig(f'trajectories_{m.name}.png')


def eval_single(automaton, name, time_limit, num_trials, save_path):
    mode = automaton.modes[name]
    env = GymEnvWrapper(
        automaton=automaton,
        mode=mode,
        transitions=automaton.transitions[name],
    )
    controller = SACController.load(
        os.path.join(save_path, f'{name}.pkl'),
    )
    state_history = []
    nonterm = 0
    normal = 0
    crash = 0
    for _ in range(num_trials):
        observation = env.reset()
        e = 0
        done = False
        while not done:
            if e > time_limit:
                break
            e += 1
            delta = controller.get_action(observation)
            state_history.append(env.state)
            observation, reward, done, info = env.step(delta)
            # print(reward)
        if e > time_limit:
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


def plot_lidar(name):
    controller = Sb3CtrlWrapper.load(f'{name}.td3', algo_name='td3')
    mode = automaton.modes[name]
    st = mode.reset()
    mode.render(st)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='evaluate controllers and mode predictor')
    ap.add_argument('--path', type=pathlib.Path, default='.',
                    help='directory in which models are stored')
    #ap.add_argument('--no-throttle', action='store_true',
    #                help='use this flag to disable throttle in the environment')
    ap.add_argument('--all', action='store_true',
                    help='use this flag to evaluate instead of specifying a list')
    ap.add_argument('--num-single-trials', type=int, default=100,
                    help='number of trials for each mode')
    ap.add_argument('--no-e2e', action='store_true',
                    help='disables end-to-end evaluation')
    ap.add_argument('--num-e2e-trials', type=int, default=20,
                    help='number of trials for end-to-end evaluation')
    ap.add_argument('--mode-length', type=int, default=100,
                    help='maximum number of time-steps in each mode')
    ap.add_argument('modes', type=str, nargs='*',
                    help='modes for which controllers will be evaluated')
    args = ap.parse_args()

    automaton = make_f110_model()
    mode_list = list(automaton.modes) if args.all else args.modes
    for name in mode_list:
        eval_single(automaton, name, args.mode_length, args.num_single_trials, args.path)
    if not args.no_e2e:
        eval_end_to_end(automaton, args.mode_length, args.num_e2e_trials, args.path)
