import os
import sys
import torch
import numpy as np
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa
from hybrid_gym import Controller
from hybrid_gym.envs.f110_turn.hybrid_env import make_f110_model
from hybrid_gym.util.io import parse_command_line_options, save_log_info
from hybrid_gym.rl.maddpg.train import MADDPG, MADDPGParams
from typing import List, Any

MAX_JUMPS = 25


if __name__ == '__main__':

    flags = parse_command_line_options()
    if not os.path.exists(flags['path']):
        os.makedirs(flags['path'])

    num_gpus = max(torch.cuda.device_count(), 1)

    automaton = make_f110_model()
    time_limits = {m: 50 for m in automaton.modes}

    # hyperparams for SAC
    params = MADDPGParams(
        max_episode_len=500,
        num_episodes=100000,
        batch_size=256,
        num_units=128,
        lr=3e-4, gamma=0.95
    )

    agent = MADDPG(automaton, params, bonus=100.)
    log_info = agent.train(time_limits, MAX_JUMPS)
    save_log_info(log_info, 'log', flags['path'])
