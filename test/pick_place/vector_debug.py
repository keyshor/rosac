# flake8: noqa
import os
import sys
import numpy as np
import gym
from stable_baselines import SAC, HER
from stable_baselines.ddpg.noise import NormalActionNoise
from stable_baselines.common.vec_env import DummyVecEnv
from gym.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
from gym.wrappers import TimeLimit
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

from hybrid_gym.envs import make_pick_place_model

automaton = make_pick_place_model(num_objects=3)


def test_vector(mode, num_samples):
    for _ in range(num_samples):
        st0 = mode.reset()
        vec0 = mode.vectorize_state(st0)
        st1 = mode.state_from_vector(vec0)
        vec1 = mode.vectorize_state(st1)
        if not np.allclose(vec0, vec1):
            print(np.abs(vec0 - vec1))
            raise Exception


if __name__ == '__main__':
    for (name, mode) in automaton.modes.items():
        print(f'testing mode {name}')
        test_vector(mode, 100)
