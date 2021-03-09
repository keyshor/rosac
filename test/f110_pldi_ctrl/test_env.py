from enum import Enum
import yaml
import matplotlib.pyplot as plt
from tensorflow.keras import models
import random
import numpy as np
import os
import sys

sys.path.append(os.path.join('..', '..'))
from hybrid_gym.envs.f110.hybrid_env import make_hybrid_env
from hybrid_gym.hybrid_env import HybridEnv
from hybrid_gym.selectors import UniformSelector, MaxJumpWrapper


class Modes(Enum):
    STRAIGHT = 'STRAIGHT'
    SQUARE_RIGHT = 'SQUARE_RIGHT'
    SQUARE_LEFT = 'SQUARE_LEFT'
    SHARP_RIGHT = 'SHARP_RIGHT'
    SHARP_LEFT = 'SHARP_LEFT'


def sigmoid(x):

    sigm = 1. / (1. + np.exp(-x))

    return sigm


def int2mode(i):
    if i == 0:
        return Modes.STRAIGHT
    elif i == 1:
        return Modes.SQUARE_RIGHT
    elif i == 2:
        return Modes.SQUARE_LEFT
    elif i == 3:
        return Modes.SHARP_RIGHT
    elif i == 4:
        return Modes.SHARP_LEFT
    else:
        raise ValueError

def str2mode(s, car_dist_f):
    if 'straight' in s or car_dist_f > 10:
        return Modes.STRAIGHT
    elif 'square_right' in s:
        return Modes.SQUARE_RIGHT
    elif 'square_left' in s:
        return Modes.SQUARE_LEFT
    elif 'sharp_right' in s:
        return Modes.SHARP_RIGHT
    elif 'sharp_left' in s:
        return Modes.SHARP_LEFT
    else:
        raise ValueError


class ComposedModePredictor:
    def __init__(self, big_file,
                 straight_file, square_right_file, square_left_file,
                 sharp_right_file, sharp_left_file, yml=False):

        self.yml = yml

        if yml:
            with open(big_file, 'rb') as f:
                self.big = yaml.full_load(f)

            self.little = {}
            with open(straight_file, 'rb') as f:
                self.little[Modes.STRAIGHT] = yaml.full_load(f)
            with open(square_right_file, 'rb') as f:
                self.little[Modes.SQUARE_RIGHT] = yaml.full_load(f)
            with open(square_left_file, 'rb') as f:
                self.little[Modes.SQUARE_LEFT] = yaml.full_load(f)
            with open(sharp_right_file, 'rb') as f:
                self.little[Modes.SHARP_RIGHT] = yaml.full_load(f)
            with open(sharp_left_file, 'rb') as f:
                self.little[Modes.SHARP_LEFT] = yaml.full_load(f)
        else:

            self.big = models.load_model(big_file)
            self.little = {
                Modes.STRAIGHT: models.load_model(straight_file),
                Modes.SQUARE_RIGHT: models.load_model(square_right_file),
                Modes.SQUARE_LEFT: models.load_model(square_left_file),
                Modes.SHARP_RIGHT: models.load_model(sharp_right_file),
                Modes.SHARP_LEFT: models.load_model(sharp_left_file)
            }
        self.current_mode = Modes.STRAIGHT

    def predict(self, observation):
        obs = observation.reshape(1, -1)

        if self.yml:
            # print(self.current_mode)
            # print(predict(self.little[self.current_mode], obs))
            if predict(self.little[self.current_mode], obs).round()[0] > 0.5:
                self.current_mode = int2mode(np.argmax(predict(self.big, obs)))
        else:
            if self.little[self.current_mode].predict(obs).round()[0] > 0.5:
                self.current_mode = int2mode(np.argmax(self.big.predict(obs)))
        return self.current_mode


class ComposedSteeringPredictor:
    def __init__(self, square_file, sharp_file, action_scale):
        with open(square_file, 'rb') as f:
            self.square_ctrl = yaml.full_load(f)
        with open(sharp_file, 'rb') as f:
            self.sharp_ctrl = yaml.full_load(f)
        self.action_scale = action_scale

    def predict(self, observation, mode):
        if mode == Modes.STRAIGHT or mode == Modes.SQUARE_RIGHT or mode == Modes.SQUARE_LEFT:
            delta = self.action_scale * predict(self.square_ctrl, observation.reshape(1, -1)).reshape((1,))
        else:
            delta = self.action_scale * predict(self.sharp_ctrl, observation.reshape(1, -1)).reshape((1,))
        if mode == Modes.SQUARE_LEFT or mode == Modes.SHARP_LEFT:
            delta = -delta
        return delta


def predict(model, inputs):
    weights = {}
    offsets = {}

    layerCount = 0
    activations = []

    for layer in range(1, len(model['weights']) + 1):

        weights[layer] = np.array(model['weights'][layer])
        offsets[layer] = np.array(model['offsets'][layer])

        layerCount += 1
        activations.append(model['activations'][layer])

    curNeurons = inputs

    for layer in range(layerCount):

        curNeurons = curNeurons.dot(weights[layer + 1].T) + offsets[layer + 1]

        if 'Sigmoid' in activations[layer]:
            curNeurons = sigmoid(curNeurons)
        elif 'Tanh' in activations[layer]:
            curNeurons = np.tanh(curNeurons)

    return curNeurons


def normalize(s):
    mean = [2.5]
    spread = [5.0]
    return (s - mean) / spread


def reverse_lidar(data):
    new_data = np.zeros((data.shape))

    for i in range(len(data)):
        new_data[i] = data[len(data) - i - 1]

    return new_data


def main(argv):

    raw_env = make_hybrid_env(straight_lengths=[10], num_lidar_rays=21)
    f110_automaton = raw_env.automaton

    env = HybridEnv(
        automaton=f110_automaton,
        selector=MaxJumpWrapper(
            wrapped_selector=UniformSelector(modes=f110_automaton.modes.values()),
            max_jumps=100
        )
    )

    state_history = {
        (m_name, pred_mode): []
        for m_name in f110_automaton.modes
        for pred_mode in Modes
    }

    mode_predictor = ComposedModePredictor(
        os.path.join('mode_pred_nn', 'big.yml'),
        os.path.join('mode_pred_nn', 'straight_little.yml'),
        os.path.join('mode_pred_nn', 'square_right_little.yml'),
        os.path.join('mode_pred_nn', 'square_left_little.yml'),
        os.path.join('mode_pred_nn', 'sharp_right_little.yml'),
        os.path.join('mode_pred_nn', 'sharp_left_little.yml'),
        yml=True
    )

    num_unsafe = 0

    action_scale = float(env.action_space.high[0])

    steering_ctrl = ComposedSteeringPredictor(
        os.path.join('steering_nn', 'tanh64x64_right_turn_lidar.yml'),
        os.path.join('steering_nn', 'tanh64x64_sharp_turn_lidar.yml'),
        action_scale
    )

    #allX = []
    #allY = []
    #allR = []
    #end_modes = set()

    #straight_pred_x = []
    #square_right_pred_x = []
    #square_left_pred_x = []
    #sharp_right_pred_x = []
    #sharp_left_pred_x = []
    #straight_pred_y = []
    #square_right_pred_y = []
    #square_left_pred_y = []
    #sharp_right_pred_y = []
    #sharp_left_pred_y = []

    observation = env.reset()

    e = 0
    done = False
    while not done:
        e += 1

        #if not state_feedback:
        observation = normalize(observation)

        mode = mode_predictor.predict(observation)
        #mode = str2mode(env.mode.name, env.state.car_dist_f)

        if mode == Modes.SQUARE_LEFT or mode == Modes.SHARP_LEFT:
            observation = reverse_lidar(observation)

        #allX.append(w.car_global_x)
        #allY.append(w.car_global_y)

        delta = steering_ctrl.predict(observation, mode)

        #if mode == Modes.STRAIGHT:
        #    straight_pred_x.append(w.car_global_x)
        #    straight_pred_y.append(w.car_global_y)
        #elif mode == Modes.SQUARE_RIGHT:
        #    square_right_pred_x.append(w.car_global_x)
        #    square_right_pred_y.append(w.car_global_y)
        #elif mode == Modes.SQUARE_LEFT:
        #    square_left_pred_x.append(w.car_global_x)
        #    square_left_pred_y.append(w.car_global_y)
        #elif mode == Modes.SHARP_RIGHT:
        #    sharp_right_pred_x.append(w.car_global_x)
        #    sharp_right_pred_y.append(w.car_global_y)
        #elif mode == Modes.SHARP_LEFT:
        #    sharp_left_pred_x.append(w.car_global_x)
        #    sharp_left_pred_y.append(w.car_global_y)
        state_history[env.mode.name, mode].append(env.state)

        observation, reward, done, info = env.step(delta)

        if not env.mode.is_safe(env.state):
            print('Crash after {} steps'.format(e))


    #end_modes.add(mode_predictor.current_mode)

    #print('number of crashes: ' + str(num_unsafe))
    # print(it)

    # plt.ylim((-1,11))
    # plt.xlim((-1.75,15.25))
    # plt.tick_params(labelsize=20)

    #plt.scatter(posX, posY, s = 1, c = 'r')
    #plt.scatter(negX, negY, s = 1, c = 'b')
    # plt.plot(negX, negY, 'b')
    for (m_name, m) in f110_automaton.modes.items():
        #fig = plt.figure(figsize=(12, 10))
        fig = plt.figure()
        m.plotHalls()
        colors = ['r', 'g', 'b', 'm', 'c']
        for (pred_mode, c) in zip(list(Modes), colors):
            x_hist = [s.car_global_x for s in state_history[m_name, pred_mode]]
            y_hist = [s.car_global_y for s in state_history[m_name, pred_mode]]
            plt.scatter(x_hist, y_hist, s=1, c=c, label=pred_mode.name)
        plt.show()
        plt.legend(markerscale=10)
        plt.savefig(f'trajectories_{m.name}.png')

    # for i in range(numTrajectories):
    #     plt.plot(allX[i], allY[i], 'r-')



if __name__ == '__main__':
    main(sys.argv[1:])
