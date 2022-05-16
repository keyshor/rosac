import os
import sys
import torch
import numpy as np
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa
from hybrid_gym import Controller
from hybrid_gym.envs.f110_turn.hybrid_env import make_f110_model
from hybrid_gym.train.reward_funcs import SVMReward, ValueBasedReward
from hybrid_gym.synthesis.abstractions import Box, StateWrapper
from hybrid_gym.train.cegrl import cegrl
from hybrid_gym.util.io import parse_command_line_options, save_log_info
from hybrid_gym.rl.ars import NNParams, ARSParams
from hybrid_gym.rl.ddpg import DDPGParams
from typing import List, Any

MAX_JUMPS = 20


class FalsifyFunc:
    '''
    Evaluation function used by the falsification algorithm.
    '''

    def __init__(self, mode):
        self.mode = mode

    def __call__(self, sass: List[Any]) -> float:
        rewards = [self.mode.reward(*sas) for sas in sass]
        return sum(rewards)


if __name__ == '__main__':

    flags = parse_command_line_options()
    if not os.path.exists(flags['path']):
        os.makedirs(flags['path'])

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(flags['gpu_num'])
    num_gpus = max(torch.cuda.device_count(), 1)

    automaton = make_f110_model()
    pre = {m: mode.get_init_pre() for m, mode in automaton.modes.items()}
    time_limits = {m: 50 for m in automaton.modes}

    # state distribution update
    num_synth_iter = 0
    if flags['synthesize']:
        num_synth_iter = MAX_JUMPS
    use_full_reset = (not flags['dagger']) and (num_synth_iter == 0)

    # reward update
    reward_funcs = None
    if flags['dynamic_rew']:
        reward_funcs = {m: ValueBasedReward(mode, automaton)
                        for m, mode in automaton.modes.items()}

    # hyperparams for SAC
    if 'NAIVE' in flags['path']:
        num_epochs = 2
    elif 'DAGGER' in flags['path']:
        num_epochs = 2
    elif 'AROSAC' in flags['path']:
        num_epochs = 1
    sac_kwargs = dict(
        hidden_dims=(64, 64),
        steps_per_epoch=10000, epochs=num_epochs,
        replay_size=50000,
        gamma=1 - 1e-2, polyak=1 - 5e-3, lr=3e-4,
        alpha=0.1,
        batch_size=256,
        start_steps=2000, update_after=2000,
        update_every=50,
        num_test_episodes=10,
        max_ep_len=50, test_ep_len=50,
        log_interval=100,
        min_alpha=0.1,
        alpha_decay=1e-2,
    )

    controllers, log_info = cegrl(automaton, pre, time_limits, num_iter=10, num_synth_iter=num_synth_iter,
                                  abstract_synth_samples=flags['abstract_samples'], print_debug=True,
                                  save_path=flags['path'], algo_name='my_sac', ensemble=flags['ensemble'],
                                  sac_kwargs=sac_kwargs, use_gpu=flags['gpu'],
                                  max_jumps=MAX_JUMPS, dagger=flags['dagger'], full_reset=use_full_reset,
                                  env_name='f110_turn', inductive_ce=flags['inductive_ce'],
                                  reward_funcs=reward_funcs)

    # save the controllers
    for (mode_name, ctrl) in controllers.items():
        ctrl.save(mode_name + '_final', flags['path'])
    save_log_info(log_info, 'log', flags['path'])
