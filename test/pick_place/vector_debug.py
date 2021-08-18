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



def test_vector(modes, num_samples):
    st_proto = modes[0].reset()
    s = np.zeros_like(modes[0].vectorize_state(st_proto))
    s_sq = np.zeros_like(s)
    max_dev = np.zeros_like(s)
    for _ in range(num_samples):
        mode = modes[np.random.choice(len(modes))]
        st0 = mode.reset()
        vec0 = mode.vectorize_state(st0)
        st1 = mode.state_from_vector(vec0)
        vec1 = mode.vectorize_state(st1)
        diff = vec1 - vec0
        s += diff
        s_sq += np.square(diff)
        max_dev = np.maximum(max_dev, np.abs(diff))
    avg = s / num_samples
    avg_sq = s_sq / num_samples
    print(avg)
    print(np.sqrt(avg_sq - np.square(avg)))
    print(max_dev)


if __name__ == '__main__':
    automaton = make_pick_place_model(num_objects=5)
    modes = list(automaton.modes.values())
    num_modes = len(modes)
    test_vector(modes, 10000)
