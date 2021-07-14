import os
import sys
import numpy as np
import gym
from stable_baselines import SAC, HER
from stable_baselines.ddpg.noise import NormalActionNoise
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.her import HERGoalEnvWrapper
from gym.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
from gym.wrappers import TimeLimit
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.train.single_mode import make_sb_model, train_stable
from hybrid_gym.util.wrappers import BaselineCtrlWrapper, DoneOnSuccessWrapper

def train_model(total_timesteps):
    env = HERGoalEnvWrapper(DoneOnSuccessWrapper(TimeLimit(
        FetchPickAndPlaceEnv(),
        max_episode_steps=50,
    )))
    model = HER(
        'MlpPolicy', env, SAC,
        gamma=0.95, buffer_size=1000000,
        ent_coef='auto', goal_selection_strategy='future',
        n_sampled_goal=4, train_freq=1, learning_starts=1000,
        verbose=0
    )
    callback = EvalCallback(
        eval_env=env, n_eval_episodes=100, eval_freq=10000,
        log_path=f'replicate_paper',
        best_model_save_path=f'replicate_paper',
    )
    model.learn(total_timesteps=total_timesteps, callback=callback)

if __name__ == '__main__':
    train_model(int(sys.argv[1]))
