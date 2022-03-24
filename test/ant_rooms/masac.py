import os
import sys
import torch
import numpy as np
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa
from hybrid_gym import Controller
from hybrid_gym.envs.ant_rooms.hybrid_env import make_ant_model
from hybrid_gym.util.io import parse_command_line_options, save_log_info
from hybrid_gym.rl.sac.masac import MaSAC
from hybrid_gym.train.reward_funcs import ValueBasedReward
from typing import List, Any

MAX_JUMPS = 5


if __name__ == '__main__':

    flags = parse_command_line_options()
    if not os.path.exists(flags['path']):
        os.makedirs(flags['path'])

    num_gpus = max(torch.cuda.device_count(), 1)

    automaton = make_ant_model()
    time_limits = {m: 500 for m in automaton.modes}

    reward_fns = {m: ValueBasedReward(mode, automaton, adversary=flags['dynamic_rew'],
                                      bonus=50.)
                  for m, mode in automaton.modes.items()}

    # hyperparams for SAC
    sac_kwargs = dict(
        hidden_dims=(256, 256),
        #steps_per_epoch=100000, epochs=3,
        steps_per_epoch=10, epochs=3,
        replay_size=1000000,
        gamma=1 - 1e-2, polyak=1 - 5e-3, lr=3e-4,
        alpha=0.1,
        batch_size=256,
        start_steps=10000, update_after=10000,
        update_every=50,
        num_test_episodes=10,
        max_ep_len=500, test_ep_len=500,
        log_interval=200,
        min_alpha=0.1,
        alpha_decay=1e-2,
        gpu_device='cuda:{}'.format(flags['gpu_num'] % num_gpus),
    )

    masac = MaSAC(automaton, 3000, time_limits, MAX_JUMPS, sac_kwargs, reward_fns,
                  verbose=True, use_gpu=flags['gpu'])
    log_info = masac.train(150)
    controllers = masac.det_controllers

    # save the controllers
    for (mode_name, ctrl) in controllers.items():
        ctrl.save(mode_name + '_final', flags['path'])
    save_log_info(log_info, 'log', flags['path'])
