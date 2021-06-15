import os
import sys
import numpy as np
sys.path.append(os.path.join('..', '..'))  # nopep8

# flake8: noqa
from hybrid_gym import Controller
from hybrid_gym.envs import make_f110_model
from hybrid_gym.synthesis.abstractions import Box, StateWrapper
from hybrid_gym.train.cegrl import cegrl


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


if __name__ == '__main__':

    f110_automaton = make_f110_model(straight_lengths=[10])

    init_vec = {m: np.array([mode.init_car_dist_s, mode.init_car_dist_f,
                             mode.init_car_heading, mode.init_car_V])
                for m, mode in f110_automaton.modes.items()}
    pre = {m: F110AbstractState(init_vec[m], init_vec[m], mode)
           for m, mode in f110_automaton.modes.items()}
    max_timesteps = {m: 100 for m in f110_automaton.modes}

    controllers = cegrl(f110_automaton, pre, max_timesteps, steps_per_iter=20000,
                        num_iter=20, action_noise_scale=4.0, verbose=1,
                        num_synth_iter=10, abstract_samples=1, print_debug=True)

    # save the controllers
    for (mode_name, ctrl) in controllers.items():
        ctrl.save(f'{mode_name}.td3')
