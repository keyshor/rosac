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
from hybrid_gym.rl.paired import Paired
from hybrid_gym.train.reward_funcs import ValueBasedReward
from typing import List, Any
from labml.configs import FloatDynamicHyperParam, IntDynamicHyperParam

MAX_JUMPS = 25


if __name__ == '__main__':

    flags = parse_command_line_options()
    if not os.path.exists(flags['path']):
        os.makedirs(flags['path'])

    num_gpus = max(torch.cuda.device_count(), 1)

    automaton = make_f110_model()
    time_limits = {m: 50 for m in automaton.modes}

    # train configs
    configs = {
        'normalize_adv': True,
        'warmup': 1024,
        'max_ep_len': 500,
        'max_steps_in_mode': 50,
        'bonus': 100.,
        'updates': 10000,
        'epochs': IntDynamicHyperParam(4),
        'batch_size': 128 * 4,
        'batches': 4,
        'value_loss_coef': FloatDynamicHyperParam(0.05),
        'entropy_bonus_coef': FloatDynamicHyperParam(0.005),
        'clip_range': FloatDynamicHyperParam(0.1),
        'learning_rate': FloatDynamicHyperParam(1e-3),
    }

    paired = Paired(automaton, 128, 'cuda:{}'.format(flags['gpu_num'] % num_gpus),
                    configs, time_limits, adversary_batch_size=8, adversary_updates=8, adv_lr=3e-3,
                    max_jumps=MAX_JUMPS, use_gpu=flags['gpu'])
    log_info = paired.train(1000)
    save_log_info(log_info, 'log', flags['path'])
