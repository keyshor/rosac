import os
import sys
import numpy as np
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa
from hybrid_gym import Controller
from hybrid_gym.envs import make_rooms_model
from hybrid_gym.envs.rooms.mode import RewardFunc
from hybrid_gym.synthesis.abstractions import Box, StateWrapper
from hybrid_gym.train.cegrl import cegrl
from hybrid_gym.util.io import parse_command_line_options, save_log_info
from hybrid_gym.rl.ars import NNParams, ARSParams
from hybrid_gym.rl.ddpg import DDPGParams
from typing import List, Any

MAX_JUMPS = 5


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

    os.environ["CUDA_VISIBLE_DEVICES"] = str(flags['gpu_num'])

    automaton = make_rooms_model()
    pre = {m: mode.get_init_pre() for m, mode in automaton.modes.items()}
    time_limits = {m: 25 for m in automaton.modes}

    # state distribution update
    num_synth_iter = 0
    if flags['synthesize']:
        num_synth_iter = MAX_JUMPS
    use_full_reset = (not flags['dagger']) and (num_synth_iter == 0)

    # reward update
    reward_funcs = None
    if flags['dynamic_rew']:
        reward_funcs = {m: RewardFunc(mode, automaton, time_limits, use_classifier=flags['falsify'],
                                      top_samples=0.25)
                        for m, mode in automaton.modes.items()}

    nn_params = NNParams(2, 2, 1.0, 128)
    ars_params = ARSParams(600, 30, 10, 0.025, 0.08, 0.95, 25, track_best=True)
    action_bound = np.ones((2,))
    ddpg_params = DDPGParams(2, 2, action_bound, actor_lr=0.001, critic_lr=0.0001, minibatch_size=128,
                             num_episodes=3000, buffer_size=200000, discount=0.95,
                             epsilon_decay=0., epsilon_min=0.1,
                             steps_per_update=100, gradients_per_update=100,
                             actor_hidden_dim=64, critic_hidden_dim=64, max_timesteps=25,
                             test_max_timesteps=25, sigma=0.15)
    sac_kwargs = dict(hidden_dims=(64, 64), episodes_per_epoch=15, max_ep_len=25, test_ep_len=25,
                      alpha=0.4)

    controllers, log_info = cegrl(automaton, pre, time_limits, num_iter=150, num_synth_iter=num_synth_iter,
                                  abstract_synth_samples=flags['abstract_samples'], print_debug=True,
                                  use_best_model=flags['best'], save_path=flags['path'], algo_name='my_sac',
                                  nn_params=nn_params, ars_params=ars_params, ddpg_params=ddpg_params,
                                  sac_kwargs=sac_kwargs, use_gpu=flags['gpu'], max_jumps=MAX_JUMPS,
                                  dagger=flags['dagger'], full_reset=use_full_reset,
                                  inductive_ce=flags['inductive_ce'], reward_funcs=reward_funcs)

    # save the controllers
    for (mode_name, ctrl) in controllers.items():
        ctrl.save(mode_name + '_final', flags['path'])
    save_log_info(log_info, 'log', flags['path'])
