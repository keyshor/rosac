import os
import sys
import torch
import numpy as np
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa
from hybrid_gym import Controller
from hybrid_gym.envs.pick_place.hybrid_env import make_pick_place_model
from hybrid_gym.train.reward_funcs import SVMReward
from hybrid_gym.synthesis.abstractions import Box, StateWrapper
from hybrid_gym.train.cegrl_mypool import cegrl_mypool
from hybrid_gym.util.io import parse_command_line_options, save_log_info
from hybrid_gym.rl.ars import NNParams, ARSParams
from hybrid_gym.rl.ddpg import DDPGParams
from typing import List, Any

MAX_JUMPS = 20
num_iter=5


if __name__ == '__main__':
    num_objects = 3

    flags = parse_command_line_options()
    if not os.path.exists(flags['path']):
        os.makedirs(flags['path'])

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(flags['gpu_num'])
    num_gpus = max(torch.cuda.device_count(), 1)

    automaton = make_pick_place_model(
        num_objects=num_objects,
        reward_type='dense',
        fixed_tower_height=True,
        flatten_obs=True,
    )
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
    mode0 = list(automaton.modes.values())[0]
    sac_kwargs = dict(
        #obs_space=mode0.observation_space, act_space=mode0.action_space,
        hidden_dims=(1024, 1024, 1024),
        #hidden_dims=(16, 16),
        steps_per_epoch=100000, epochs=3,
        #steps_per_epoch=10, epochs=2,
        replay_size=1000000,
        #replay_size=100,
        gamma=1 - 5e-2, polyak=1 - 5e-3, lr=1e-3,
        alpha=0.1,
        batch_size=256,
        start_steps=10000, update_after=10000,
        update_every=50,
        num_test_episodes=10,
        max_ep_len=20, test_ep_len=20,
        log_interval=10000,
        min_alpha=0.1,
        alpha_decay=1e-2,
    )

    controllers, log_info = cegrl_mypool(
            automaton, pre, time_limits,
            mode_groups=[
                [automaton.modes[f'PICK_OBJ_PT1_{i}'] for i in range(num_objects)],
                [automaton.modes[f'PICK_OBJ_PT2_{i}'] for i in range(num_objects)],
                [automaton.modes[f'PICK_OBJ_PT3_{i}'] for i in range(num_objects)],
                [automaton.modes[f'PLACE_OBJ_PT1_{i}'] for i in range(num_objects)],
                [automaton.modes[f'PLACE_OBJ_PT2_{i}'] for i in range(num_objects)],
            ] + [
                [automaton.modes[f'MOVE_WITHOUT_OBJ_{i}']] for i in range(num_objects)
            ] + [
                [automaton.modes[f'MOVE_WITH_OBJ_h{j}_{i}']]
                for i in range(num_objects)
                for j in range(num_objects)
            ],
            num_iter=round(num_iter / flags['ensemble']),
            num_synth_iter=num_synth_iter,
            abstract_synth_samples=flags['abstract_samples'], print_debug=True,
            save_path=flags['path'], algo_name='my_sac', ensemble=flags['ensemble'],
            ars_kwargs=ars_kwargs, sac_kwargs=sac_kwargs, use_gpu=flags['gpu'],
            max_jumps=MAX_JUMPS, dagger=flags['dagger'], full_reset=use_full_reset,
            env_name='pick_place', inductive_ce=flags['inductive_ce'],
            reward_funcs=reward_funcs, max_processes=6)

    # save the controllers
    for (mode_name, ctrl) in controllers.items():
        ctrl.save(mode_name + '_final', flags['path'])
    save_log_info(log_info, 'log', flags['path'])
