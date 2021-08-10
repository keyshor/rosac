import os
import sys
import numpy as np
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa
from hybrid_gym import Controller
from hybrid_gym.envs import make_f110_model
from hybrid_gym.synthesis.abstractions import Box, StateWrapper
from hybrid_gym.train.cegrl import cegrl
from hybrid_gym.util.io import parse_command_line_options
from typing import List, Any


class F110AbstractState(StateWrapper):
    '''
    low, high: [x, y, theta, v]
    '''

    def __init__(self, low, high, mode):
        super().__init__(mode, Box(low, high))

    def __str__(self):
        low = self.abstract_state.low
        high = self.abstract_state.high
        return 'x in [{}, {}]\n'.format(low[0], high[0]) + \
            'y in [{}, {}]\n'.format(low[1], high[1]) + \
            'theta in [{}, {}]\n'.format(low[2], high[2]) + \
            'v in [{}, {}]'.format(low[3], high[3])


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

    f110_automaton = make_f110_model(straight_lengths=[10], simple=flags['simple'],
                                     use_throttle=(not flags['no_throttle']))

    init_vec = {m: np.array([mode.init_car_dist_s, mode.init_car_dist_f,
                             mode.init_car_heading, mode.init_car_V])
                for m, mode in f110_automaton.modes.items()}
    delta = np.array([0.05, 0.05, 0.001, 0.1])
    pre = {m: F110AbstractState(init_vec[m] - delta, init_vec[m] + delta, mode)
           for m, mode in f110_automaton.modes.items()}
    time_limits = {m: 100 for m in f110_automaton.modes}

    falsify_func = None
    if flags['falsify']:
        falsify_func = {name: FalsifyFunc(mode) for name, mode in f110_automaton.modes.items()}

    controllers = cegrl(f110_automaton, pre, time_limits, steps_per_iter=100000,
                        num_iter=10, num_synth_iter=10, abstract_samples=20, print_debug=True,
                        action_noise_scale=4.0, verbose=1, use_best_model=(not flags['no_best']),
                        falsify_func=falsify_func, save_path=flags['path'])

    # save the controllers
    for (mode_name, ctrl) in controllers.items():
        ctrl.save(os.path.join(flags['path'], mode_name + '.td3'))
