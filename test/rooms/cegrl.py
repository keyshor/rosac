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
        num_synth_iter = 15
    if flags['synthesize']:
        num_synth_iter = 15

    nn_params = NNParams(2, 2, 1.0, 32)
    ars_params = ARSParams(200, 30, 15, 0.05, 0.3, 0.95, 25)

    controllers, log_info = cegrl(automaton, pre, time_limits, num_iter=300, num_synth_iter=num_synth_iter,
                                  abstract_synth_samples=flags['abstract_samples'], print_debug=True,
                                  use_best_model=flags['best'], falsify_func=falsify_func,
                                  save_path=flags['path'], algo_name='ars', nn_params=nn_params,
                                  ars_params=ars_params, use_gpu=flags['gpu'], max_jumps=15,
                                  dagger=flags['dagger'], full_reset=(num_synth_iter == 0))

    # save the controllers
    for (mode_name, ctrl) in controllers.items():
        ctrl.save(mode_name + '_final', flags['path'])
    save_log_info(log_info, 'log', flags['path'])
