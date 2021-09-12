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
from hybrid_gym.util.io import parse_command_line_options
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

    init_vec = {m: np.zeros((4,)) for m in automaton.modes}
    delta = np.ones((4,))
    pre = {m: StateWrapper(mode, Box(low=init_vec[m] - delta, high=init_vec[m] + delta))
           for m, mode in automaton.modes.items()}
    time_limits = {m: 25 for m in automaton.modes}

    falsify_func = None
    if flags['falsify']:
        falsify_func = {name: FalsifyFunc(mode) for name, mode in automaton.modes.items()}

    controllers = cegrl(automaton, pre, time_limits, steps_per_iter=50000,
                        num_iter=15, num_synth_iter=15, abstract_synth_samples=flags['abstract_samples'],
                        print_debug=True, batch_size=256, action_noise_scale=0.2, verbose=1,
                        learning_rate=0.0003, tau=0.001, buffer_size=50000,
                        # train_kwargs={'eval_freq': 1000, 'n_eval_episodes': 10},
                        use_best_model=(not flags['no_best']), policy_kwargs={'net_arch': [32, 32]},
                        falsify_func=falsify_func, save_path=flags['path'], algo_name='td3')

    # save the controllers
    for (mode_name, ctrl) in controllers.items():
        ctrl.save(os.path.join(flags['path'], mode_name + '.td3'))
