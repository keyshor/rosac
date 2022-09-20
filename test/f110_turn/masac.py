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
from hybrid_gym.rl.sac.masac import MaSAC
from hybrid_gym.train.reward_funcs import ValueBasedReward
from typing import List, Any

MAX_JUMPS = 25


if __name__ == '__main__':

    flags = parse_command_line_options()
    if not os.path.exists(flags['path']):
        os.makedirs(flags['path'])

    num_gpus = max(torch.cuda.device_count(), 1)

    automaton = make_f110_model()
    time_limits = {m: 50 for m in automaton.modes}

    reward_fns = {m: ValueBasedReward(mode, automaton, adversary=flags['dynamic_rew'],
                                      bonus=50.)
                  for m, mode in automaton.modes.items()}

    # hyperparams for SAC
    sac_kwargs = dict(
        hidden_dims=(128, 128),
        steps_per_epoch=500, epochs=200,
        alpha=0.06, min_alpha=0.03, alpha_decay=0.0003,
        lr=1e-3,
        batch_size=128,
        num_test_episodes=10,
        max_ep_len=50, test_ep_len=50,
        log_interval=100
    )

    masac = MaSAC(automaton, 1250, time_limits, MAX_JUMPS, sac_kwargs, reward_fns,
                  verbose=True, use_gpu=flags['gpu'])
    log_info = masac.train(2000000)
    controllers = masac.det_controllers

    # save the controllers
    for (mode_name, ctrl) in controllers.items():
        ctrl.save(mode_name + '_final', flags['path'])
    save_log_info(log_info, 'log', flags['path'])
