import os
import sys
import argparse
import pathlib
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.util.wrappers import Sb3CtrlWrapper, GymEnvWrapper
from hybrid_gym.train.mode_pred import ScipyModePredictor
from hybrid_gym.envs import make_rooms_model
from hybrid_gym.hybrid_env import HybridEnv
from hybrid_gym.selectors import UniformSelector, MaxJumpWrapper

import matplotlib.pyplot as plt


def eval_end_to_end(automaton, max_steps_in_mode, num_trials, save_path):

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
            env=env,
        )
        for name in automaton.modes
    }

    num_normal = 0
    num_stuck = 0
    num_crash = 0

    for _ in range(num_trials):
        observation = env.reset()
        e = 0
        e_in_mode = 0
        done = False
        while not done and e_in_mode < max_steps_in_mode:
            e += 1
            e_in_mode += 1
            mode = env.mode.name
            action = controllers[mode].get_action(observation)
            observation, reward, done, info = env.step(action)
            if info['jump']:
                e_in_mode = 0
        if e_in_mode >= max_steps_in_mode:
            num_stuck += 1
            print(f'stuck in mode {mode} after {e} steps')
        elif env.mode.is_safe(env.state):
            num_normal += 1
            print(f'terminated normally after {e} steps')
        else:
            num_crash += 1
            print(f'crash after {e} steps in mode {mode}')
    print(f'normal {num_normal}, stuck {num_stuck}, crash {num_crash}')


def eval_single(automaton, name, time_limit, num_trials, save_path):
    mode = automaton.modes[name]
    env = GymEnvWrapper(mode, automaton.transitions[name])
    controller = Sb3CtrlWrapper.load(
        os.path.join(save_path, name, 'best_model.zip'),
        algo_name='td3',
        env=env,
    )
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
            action = controller.get_action(observation)
            observation, reward, done, info = env.step(action)
            # print(reward)
        if e > time_limit:
            #print(f'spent more than {time_limit} steps in the mode')
            nonterm += 1
        elif env.mode.is_safe(env.state):
            #print(f'terminated normally after {e} steps')
            normal += 1
        else:
            #print(f'crash after {e} steps')
            crash += 1
    print(f'{name}: nonterm = {nonterm}, normal = {normal}, crash = {crash}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='evaluate controllers and mode predictor')
    ap.add_argument('--path', type=pathlib.Path, default='.',
                    help='directory in which models are stored')
    ap.add_argument('--all', action='store_true',
                    help='use this flag to evaluate instead of specifying a list')
    ap.add_argument('--num-single-trials', type=int, default=1000,
                    help='number of trials for each mode')
    ap.add_argument('--no-e2e', action='store_true',
                    help='disables end-to-end evaluation')
    ap.add_argument('--num-e2e-trials', type=int, default=20,
                    help='number of trials for end-to-end evaluation')
    ap.add_argument('--mode-length', type=int, default=25,
                    help='maximum number of time-steps in each mode')
    ap.add_argument('modes', type=str, nargs='*',
                    help='modes for which controllers will be evaluated')
    args = ap.parse_args()

    automaton = make_rooms_model()
    mode_list = list(automaton.modes) if args.all else args.modes
    for name in mode_list:
        eval_single(automaton, name, args.mode_length, args.num_single_trials, args.path)
    if not args.no_e2e:
        eval_end_to_end(automaton, args.mode_length, args.num_e2e_trials, args.path)
