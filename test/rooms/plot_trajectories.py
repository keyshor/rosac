import os
import sys
import argparse
import pathlib
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.util.wrappers import BaselineCtrlWrapper, GymEnvWrapper
from hybrid_gym.rl.ars import NNPolicy
from hybrid_gym.train.mode_pred import ScipyModePredictor
from hybrid_gym.envs.rooms.hybrid_env import make_rooms_model
from hybrid_gym.hybrid_env import HybridEnv
from hybrid_gym.selectors import UniformSelector, MaxJumpWrapper
from hybrid_gym.util.test import get_rollout
from hybrid_gym.util.io import parse_command_line_options

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def plot_trajectories(automaton, name, path):
    mode = automaton.modes[name]
    transitions = automaton.transitions[name]
    controller = NNPolicy.load(name + '_final', path)
    state_history = []
    start_history = []
    end_history = []
    nonterm = 0
    normal = 0
    crash = 0
    for _ in range(100):
        sass, info = get_rollout(mode, transitions, controller, max_timesteps=25)
        start_history.append(sass[0][0][0])
        end_history.append(sass[-1][-1][-1])
        state_history.append([s[1] for s, _, _ in sass] + [sass[-1][-1][-1]])
        if not info['safe']:
            crash += 1
        elif info['jump'] is None:
            nonterm += 1
        else:
            normal += 1

    mode.grid_params.plot_room()
    state_history = np.concatenate(state_history) + mode.grid_params.full_size/2
    start_history = np.array(start_history) + mode.grid_params.full_size/2
    end_history = np.array(end_history) + mode.grid_params.full_size/2
    plt.scatter(state_history[:, 0], state_history[:, 1],
                s=1, label='trajectory', color='blue')
    plt.scatter(start_history[:, 0], start_history[:, 1], s=1, label='start', color='orange')
    plt.scatter(end_history[:, 0], end_history[:, 1], s=1, label='end', color='green')
    print(f'{name}: nonterm = {nonterm}, normal = {normal}, crash = {crash}')
    plt.show()


if __name__ == '__main__':
    flags = parse_command_line_options()
    automaton = make_rooms_model()
    plot_trajectories(automaton, 'left', flags['path'])
