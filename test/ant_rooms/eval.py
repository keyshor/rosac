import os
import sys
import argparse
import pathlib
import pickle
import random
import numpy as np
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.util.wrappers import GymEnvWrapper
from hybrid_gym.envs.ant_rooms.hybrid_env import make_ant_model
from hybrid_gym.rl.sac.sac import SACController
from hybrid_gym.hybrid_env import HybridEnv
from hybrid_gym.selectors import UniformSelector, MaxJumpWrapper

import matplotlib.pyplot as plt
import matplotlib


def eval_end_to_end(automaton, max_steps_in_mode, num_trials, save_path, render):

    env = HybridEnv(
        automaton=automaton,
        selector=MaxJumpWrapper(
            wrapped_selector=UniformSelector(modes=automaton.modes.values()),
            max_jumps=100
        ),
        max_timesteps=1000000000000000000000000,
        max_timesteps_per_mode=max_steps_in_mode,
    )

    controllers = {
        name: SACController.load(
            os.path.join(save_path, f'{name}_final.pkl'),
        )
        for name in automaton.modes
    }

    with open(os.path.join(save_path, 'data.pkl'), 'rb') as fh:
        start_sts = pickle.load(fh)
    def reset():
        env.reset()
        env.state = random.choice(start_sts)
        return env.observe()

    num_normal = 0
    num_stuck = 0
    num_crash = 0
    survival_times = []

    for trial_index in range(num_trials):
        if render:
            vid_path = os.path.join(save_path, f'vid_e2e_{trial_index:0>4}')
            os.makedirs(vid_path, exist_ok=True)
        observation = reset()
        num_jumps = 0
        done = False
        frame_index = 0
        while not done:
            action = controllers[env.mode.name].get_action(observation)
            observation, reward, done, info = env.step(action)
            if render:
                matplotlib.image.imsave(
                    os.path.join(
                        vid_path,
                        f'frame_{frame_index:0>5}.png',
                    ),
                    np.flip(env.mode.ant.sim.render(1280, 720), axis=0),
                    vmin=0, vmax=255,
                )
            if info['jump']:
                num_jumps += 1
            frame_index += 1
        if info['stuck']:
            num_stuck += 1
            print(f'stuck in mode {env.mode.name} after {num_jumps} jumps')
        elif env.mode.is_safe(env.state):
            num_normal += 1
            print(f'terminated normally after {num_jumps} jumps')
        else:
            num_crash += 1
            print(f'crash after {num_jumps} jumps in mode {env.mode.name}')
        survival_times.append(num_jumps)
    avg_survival = sum(survival_times) / len(survival_times)
    print(f'normal {num_normal}, stuck {num_stuck}, crash {num_crash}, survival {avg_survival}')


def eval_single(automaton, name, time_limit, num_trials, save_path, render):
    mode = automaton.modes[name]
    env = GymEnvWrapper(automaton, mode, automaton.transitions[name])
    controller = SACController.load(
        os.path.join(save_path, f'{name}_final.pkl'),
    )
    with open(os.path.join(save_path, 'data.pkl'), 'rb') as fh:
        start_sts = pickle.load(fh)
    def reset():
        env.state = random.choice(start_sts)
        return env.observe()
    nonterm = 0
    normal = 0
    crash = 0
    min_success = time_limit + 1
    max_success = -1
    for trial_index in range(num_trials):
        if render:
            vid_path = os.path.join(save_path, f'vid_{name}_{trial_index:0>4}')
            os.makedirs(vid_path, exist_ok=True)
        observation = reset()
        e = 0
        done = False
        while not done:
            if e > time_limit:
                break
            action = controller.get_action(observation)
            observation, reward, done, info = env.step(action)
            if render:
                matplotlib.image.imsave(
                    os.path.join(
                        vid_path,
                        f'frame_{e:0>5}.png',
                    ),
                    np.flip(env.mode.ant.sim.render(1280, 720), axis=0),
                    vmin=0, vmax=255,
                )
            # print(reward)
            e += 1
        if e > time_limit:
            #print(f'spent more than {time_limit} steps in the mode')
            nonterm += 1
        elif env.mode.is_safe(env.state):
            #print(f'terminated normally after {e} steps')
            normal += 1
            min_success = min(e, min_success)
            max_success = max(e, max_success)
        else:
            #print(f'crash after {e} steps')
            crash += 1
    print(f'{name}: nonterm = {nonterm}, normal = {normal}, crash = {crash}, success min = {min_success}, max = {max_success}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='evaluate controllers and mode predictor')
    ap.add_argument('--path', type=pathlib.Path, default='.',
                    help='directory in which models are stored')
    ap.add_argument('--no-throttle', action='store_true',
                    help='use this flag to disable throttle in the environment')
    ap.add_argument('--all', action='store_true',
                    help='use this flag to evaluate instead of specifying a list')
    ap.add_argument('--num-single-trials', type=int, default=10,
                    help='number of trials for each mode')
    ap.add_argument('--num-e2e-trials', type=int, default=20,
                    help='number of trials for end-to-end evaluation')
    ap.add_argument('--mode-length', type=int, default=1000,
                    help='maximum number of time-steps in each mode')
    ap.add_argument('--render', action='store_true',
                    help='use this flag to render the simultion as images')
    ap.add_argument('modes', type=str, nargs='*',
                    help='modes for which controllers will be evaluated')
    args = ap.parse_args()

    automaton = make_ant_model()
    mode_list = list(automaton.modes) if args.all else args.modes
    for name in mode_list:
        eval_single(automaton, name, args.mode_length, args.num_single_trials, args.path, args.render)
    if args.num_e2e_trials > 0:
        eval_end_to_end(automaton, args.mode_length, args.num_e2e_trials, args.path, args.render)
