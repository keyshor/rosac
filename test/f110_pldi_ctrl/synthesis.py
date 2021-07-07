import os
import sys
sys.path.append(os.path.join('..', '..'))  # nopep8

# flake8: noqa: E402
from test_env import (ComposedSteeringPredictor, ComposedModePredictor,
                      normalize, reverse_lidar, Modes)
from hybrid_gym import Controller
from hybrid_gym.envs import make_f110_model
from hybrid_gym.selectors import UniformSelector, MaxJumpWrapper
from hybrid_gym.synthesis.abstractions import AbstractState, Box, StateWrapper
from hybrid_gym.synthesis.ice import synthesize
from hybrid_gym.falsification.single_mode import falsify, reward_eval_func
from hybrid_gym.util.test import end_to_end_test
from copy import deepcopy

import numpy as np


class FullController(Controller):
    def __init__(self, mode_predictor, steering_ctrl):
        self.mode_predictor = mode_predictor
        self.steering_ctrl = steering_ctrl
        self.reset()

    def get_action(self, observation):
        observation = normalize(observation)
        mode = self.mode_predictor.predict(observation)
        if mode == Modes.SQUARE_LEFT or mode == Modes.SHARP_LEFT:
            observation = reverse_lidar(observation)
        delta = self.steering_ctrl.predict(observation, mode)
        return delta

    def reset(self):
        self.mode_predictor.current_mode = Modes.STRAIGHT


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

    f110_automaton = make_f110_model(straight_lengths=[10], num_lidar_rays=21)

    mode_predictor = ComposedModePredictor(
        os.path.join('mode_pred_nn', 'big.yml'),
        os.path.join('mode_pred_nn', 'straight_little.yml'),
        os.path.join('mode_pred_nn', 'square_right_little.yml'),
        os.path.join('mode_pred_nn', 'square_left_little.yml'),
        os.path.join('mode_pred_nn', 'sharp_right_little.yml'),
        os.path.join('mode_pred_nn', 'sharp_left_little.yml'),
        yml=True
    )

    action_scale = float(f110_automaton.action_space.high[0])

    steering_ctrl = ComposedSteeringPredictor(
        os.path.join('steering_nn', 'tanh64x64_right_turn_lidar.yml'),
        os.path.join('steering_nn', 'tanh64x64_sharp_turn_lidar.yml'),
        action_scale
    )

    selector = MaxJumpWrapper(
        wrapped_selector=UniformSelector(modes=f110_automaton.modes.values()),
        max_jumps=5
    )

    controllers = {m: FullController(mode_predictor, steering_ctrl) for m in f110_automaton.modes}
    init_vec = {m: np.array([mode.init_car_dist_s, mode.init_car_dist_f,
                             mode.init_car_heading, mode.init_car_V])
                for m, mode in f110_automaton.modes.items()}
    pre = {m: F110AbstractState(init_vec[m], init_vec[m], mode)
           for m, mode in f110_automaton.modes.items()}
    time_limits = {m: 100 for m in f110_automaton.modes}

    end_to_end_test(f110_automaton, selector, FullController(mode_predictor, steering_ctrl),
                    time_limits, num_rollouts=10, print_debug=True)

    ces = synthesize(f110_automaton, controllers, pre, time_limits, 10, 20, 1, True)
    mname = 'f110_square_right'
    worst_states = falsify(f110_automaton.modes[mname], f110_automaton.transitions[mname],
                           controllers[mname], pre[mname],
                           reward_eval_func(f110_automaton.modes[mname]),
                           100, 100, 20, 5, alpha=0.6, print_debug=True)
