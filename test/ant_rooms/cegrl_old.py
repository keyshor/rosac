import os
import sys
import torch
import numpy as np
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa
from hybrid_gym import Controller
from hybrid_gym.envs.ant_rooms.hybrid_env import make_ant_model
from hybrid_gym.train.reward_funcs import SVMReward
from hybrid_gym.synthesis.abstractions import Box, StateWrapper
from hybrid_gym.train.cegrl_mypool import cegrl_mypool
from hybrid_gym.util.io import parse_command_line_options, save_log_info
from hybrid_gym.rl.ars import NNParams, ARSParams
from hybrid_gym.rl.ddpg import DDPGParams
from typing import List, Any

MAX_JUMPS = 5


if __name__ == '__main__':

    flags = parse_command_line_options()
    if not os.path.exists(flags['path']):
        os.makedirs(flags['path'])

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(flags['gpu_num'])
    num_gpus = max(torch.cuda.device_count(), 1)

    automaton = make_ant_model()
    pre = {m: mode.get_init_pre() for m, mode in automaton.modes.items()}
    time_limits = {m: 500 for m in automaton.modes}

    # state distribution update
    num_synth_iter = 0
    if flags['synthesize']:
        num_synth_iter = MAX_JUMPS
    use_full_reset = (not flags['dagger']) and (num_synth_iter == 0)

    # reward update
    reward_funcs = None
    if flags['dynamic_rew']:
        reward_funcs = {m: SVMReward(mode, automaton, time_limits)
                        for m, mode in automaton.modes.items()}

     # hyperparams for ARS
    nn_params = NNParams(2, 2, 1.0, 128)
    ars_params = ARSParams(600, 30, 10, 0.025, 0.08, 0.95, 25, track_best=True)
    ars_kwargs = dict(nn_params=nn_params, ars_params=ars_params)

    # hyperparams for SAC
    #mode0 = list(automaton.modes.values())[0]
    sac_kwargs = dict(
        #obs_space=mode0.observation_space, act_space=mode0.action_space,
        hidden_dims=(256, 256),
        steps_per_epoch=100000, epochs=3,
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
    )

    controllers, log_info = cegrl_mypool(automaton, pre, time_limits, num_iter=round(5 / flags['ensemble']),
                                  num_synth_iter=num_synth_iter,
                                  abstract_synth_samples=flags['abstract_samples'], print_debug=True,
                                  save_path=flags['path'], algo_name='my_sac', ensemble=flags['ensemble'],
                                  ars_kwargs=ars_kwargs, sac_kwargs=sac_kwargs, use_gpu=flags['gpu'],
                                  max_jumps=MAX_JUMPS, dagger=flags['dagger'], full_reset=use_full_reset,
                                  env_name='ant_rooms', inductive_ce=flags['inductive_ce'],
                                  reward_funcs=reward_funcs,
                                  max_processes=5)

    # save the controllers
    for (mode_name, ctrl) in controllers.items():
        ctrl.save(mode_name + '_final', flags['path'])
    save_log_info(log_info, 'log', flags['path'])
