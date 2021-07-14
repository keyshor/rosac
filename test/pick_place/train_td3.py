import os
import sys
import numpy as np
import gym
from stable_baselines import SAC, HER
from stable_baselines.ddpg.noise import NormalActionNoise
from stable_baselines.td3.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from gym.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
from gym.wrappers import TimeLimit
sys.path.append(os.path.join('..', '..'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.train.single_mode import make_sb_model, train_stable, BaselineCtrlWrapper
from hybrid_gym.train.mode_pred import train_mode_predictor
from hybrid_gym.envs import make_pick_place_model
from hybrid_gym.envs.pick_place.mode import PickPlaceMode
from hybrid_gym.train.rlbl_zoo_utils.wrappers import DoneOnSuccessWrapper

class CustomTd3Policy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         **kwargs,
                         layers=[256],
                         layer_norm=False,
                         feature_extraction='mlp')

if __name__ == '__main__':
    automaton = make_pick_place_model(num_objects=3, reward_type='dense')
    if len(sys.argv) >= 2:
        mode_list = [(name, automaton.modes[name]) for name in sys.argv[1:]]
    else:
        mode_list = list(automaton.modes.items())
    for (name, mode) in mode_list:
        model = make_sb_model(
            mode,
            automaton.transitions[name],
            algo_name='td3',
            custom_policy=CustomTd3Policy,
            gamma=0.95,
            action_noise_scale=0.15,
            buffer_size=200000,
            verbose=2
        )
        train_stable(model, mode, automaton.transitions[name],
                     total_timesteps=50000, algo_name='td3')
        ctrl = BaselineCtrlWrapper(model)
        ctrl.save(f'{name}.td3')
