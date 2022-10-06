import os
import sys
import torch
import numpy as np
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa
from hybrid_gym import Controller
from hybrid_gym.envs import make_rooms_model
from hybrid_gym.util.io import parse_command_line_options, save_log_info
from hybrid_gym.rl.paired import Paired
from hybrid_gym.train.reward_funcs import ValueBasedReward
from typing import List, Any
from labml.configs import FloatDynamicHyperParam, IntDynamicHyperParam

MAX_JUMPS = 5


if __name__ == '__main__':

    flags = parse_command_line_options()
    if not os.path.exists(flags['path']):
        os.makedirs(flags['path'])

    num_gpus = max(torch.cuda.device_count(), 1)

    automaton = make_rooms_model()
    time_limits = {m: 25 for m in automaton.modes}

    # train configs
    configs = {
        'normalize_adv': False,
        'warmup': 1024,
        'max_ep_len': 150,
        'max_steps_in_mode': 25,
        'bonus': 25.,
        'updates': 10000,
        'epochs': IntDynamicHyperParam(8),
        'batch_size': 128 * 8,
        'batches': 4,
        'value_loss_coef': FloatDynamicHyperParam(1.0),
        'entropy_bonus_coef': FloatDynamicHyperParam(0.005),
        'clip_range': FloatDynamicHyperParam(0.2),
        'learning_rate': FloatDynamicHyperParam(1e-2),
    }

    paired = Paired(automaton, 64, 'cuda:{}'.format(flags['gpu_num'] % num_gpus),
                    configs, time_limits, adversary_batch_size=8, adversary_updates=8,
                    max_jumps=MAX_JUMPS, use_gpu=flags['gpu'])
    log_info = paired.train(500)
    save_log_info(log_info, 'log', flags['path'])
