import os
import sys
sys.path.append(os.path.join('..', '..'))  # nopep8

from test_env import (ComposedSteeringPredictor, ComposedModePredictor,
                      normalize, reverse_lidar, Modes)
from hybrid_gym import Controller
from hybrid_gym.envs import make_f110_model
from hybrid_gym.synthesis.abstractions import AbstractState, Box
from hybrid_gym.synthesis.ice import synthesize
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


class F110AbstractState(AbstractState):
    '''
    low, high: [x, y, theta, v]
    '''

    def __init__(self, low, high, mode):
        self.box = Box(low, high)
        self.mode = mode
        self.init_state = mode.reset()

    def contains(self, state):
        return self.box.contains(self.get_array(state))

    def extend(self, state):
        self.box.extend(self.get_array(state))

    def sample(self):
        return self.get_state(self.box.sample())

    def get_array(self, state):
        return np.array([state.car_dist_s, state.car_dist_f, state.car_heading, state.car_V])

    def get_state(self, arr):
        return self.mode.set_state_local(x=arr[0], y=arr[1], theta=arr[2], v=arr[3], old_st=self.init_state)

    def __str__(self):
        low = self.box.low
        high = self.box.high
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

    controllers = {m: FullController(mode_predictor, steering_ctrl) for m in f110_automaton.modes}
    init_vec = {m: np.array([mode.init_car_dist_s, mode.init_car_dist_f,
                             mode.init_car_heading, mode.init_car_V])
                for m, mode in f110_automaton.modes.items()}
    pre = {m: F110AbstractState(init_vec[m], init_vec[m], mode)
           for m, mode in f110_automaton.modes.items()}
    max_timesteps = {m: 100 for m in f110_automaton.modes}

    ces = synthesize(f110_automaton, controllers, pre, max_timesteps, 5, 1, 0, True)
