import os
import sys
import numpy as np
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa
from hybrid_gym import Controller
from hybrid_gym.envs import make_rooms_model
from hybrid_gym.synthesis.abstractions import Box, StateWrapper
from hybrid_gym.train.cegrl import cegrl
from hybrid_gym.util.io import parse_command_line_options, save_log_info
from hybrid_gym.rl.ars import NNParams, ARSParams
from hybrid_gym.rl.ddpg import DDPGParams
from typing import List, Any


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

    automaton = make_rooms_model()
    pre = {m: mode.get_init_pre() for m, mode in automaton.modes.items()}
    time_limits = {m: 25 for m in automaton.modes}

    falsify_func = None
    num_synth_iter = 0
    if flags['falsify']:
        falsify_func = {name: FalsifyFunc(mode) for name, mode in automaton.modes.items()}
        num_synth_iter = 5
    if flags['synthesize']:
        num_synth_iter = 5
    use_full_reset = (not flags['dagger']) and (num_synth_iter == 0)

    nn_params = NNParams(2, 2, 1.0, 32)
    ars_params = ARSParams(500, 30, 15, 0.05, 0.3, 0.95, 25)
    action_bound = np.ones((2,))
    ddpg_params = DDPGParams(2, 2, action_bound, actor_lr=0.01, critic_lr=0.003, minibatch_size=64,
                             num_episodes=5000, buffer_size=100000, discount=0.95,
                             epsilon_decay=3e-7, epsilon_min=0.1,
                             steps_per_update=100, gradients_per_update=100,
                             actor_hidden_dim=32, critic_hidden_dim=32, max_timesteps=25,
                             test_max_timesteps=25, sigma=0.2)

    controllers, log_info = cegrl(automaton, pre, time_limits, num_iter=500, num_synth_iter=num_synth_iter,
                                  abstract_synth_samples=flags['abstract_samples'], print_debug=True,
                                  use_best_model=flags['best'], falsify_func=falsify_func,
                                  save_path=flags['path'], algo_name='my_ddpg', nn_params=nn_params,
                                  ars_params=ars_params, ddpg_params=ddpg_params, use_gpu=flags['gpu'],
                                  max_jumps=5, dagger=flags['dagger'], full_reset=use_full_reset,
                                  inductive_ce=flags['inductive_ce'])

    # save the controllers
    for (mode_name, ctrl) in controllers.items():
        ctrl.save(mode_name + '_final', flags['path'])
    save_log_info(log_info, 'log', flags['path'])
