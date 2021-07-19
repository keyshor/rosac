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

# flake8: noqa: E402
from hybrid_gym.train.single_mode import make_sb_model, train_stable
from hybrid_gym.util.wrappers import BaselineCtrlWrapper, DoneOnSuccessWrapper
from hybrid_gym.train.mode_pred import train_mode_predictor
from hybrid_gym.envs import make_pick_place_model
from hybrid_gym.envs.pick_place.mode import PickPlaceMode

automaton = make_pick_place_model(num_objects=3)


def train_single(name, total_timesteps):
    mode = automaton.modes[name]
    model = make_sb_model(
        mode,
        automaton.transitions[name],
        algo_name='her',
        wrapped_algo='sac',
        gamma=0.95, buffer_size=1000000,
        ent_coef='auto', goal_selection_strategy='future',
        n_sampled_goal=4, train_freq=1, learning_starts=1000,
        verbose=2
    )
    train_stable(model, mode, automaton.transitions[name],
                 total_timesteps=total_timesteps, algo_name='her')
    ctrl = BaselineCtrlWrapper(model)


if __name__ == '__main__':
    train_single(sys.argv[1], int(sys.argv[2]))
