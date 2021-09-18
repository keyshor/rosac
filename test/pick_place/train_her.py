import os
import sys
import argparse
import pathlib
import numpy as np
import gym
from stable_baselines import SAC, HER
from stable_baselines.ddpg.noise import NormalActionNoise
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines3 import HerReplayBuffer
from gym.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
from gym.wrappers import TimeLimit
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.train.single_mode import make_sb3_model, train_sb3
from hybrid_gym.util.wrappers import Sb3CtrlWrapper, DoneOnSuccessWrapper
from hybrid_gym.train.mode_pred import train_mode_predictor
from hybrid_gym.envs import make_pick_place_model
from hybrid_gym.envs.pick_place.mode import PickPlaceMode, ModeType


def train_single(automaton, names, total_timesteps, save_path, model_path):
    modes = [automaton.modes[n] for n in names]
    mode_info = [(automaton.modes[n], automaton.transitions[n], None, None)
                 for n in names]
    model = make_sb3_model(
        mode_info,
        algo_name='tqc',
        policy='MultiInputPolicy',
        gamma=0.95, buffer_size=1000000,
        batch_size=2048,
        learning_rate=0.001,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy='future',
            online_sampling=True,
            max_episode_length=50,
        ),
        policy_kwargs=dict(
            net_arch=[512, 512, 512],
            n_critics=2,
        ),
        reward_offset=0.0,
        is_goal_env=True,
        verbose=0,
    )
    train_sb3(model, mode_info,
              total_timesteps=total_timesteps, algo_name='her',
              reward_offset=0.0, is_goal_env=True,
              save_path=save_path, custom_best_model_path=model_path)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='train controllers and mode predictor')
    ap.add_argument('--path', type=pathlib.Path, default='.',
                    help='directory in which models will be saved')
    ap.add_argument('--num-objects', type=int, default=3,
                    help='number of objects')
    ap.add_argument('--timesteps', type=int, default=500000,
                    help='number of timesteps to train each controller')
    ap.add_argument('--all', action='store_true',
                    help='use this flag to train all modes instead of specifying a list')
    ap.add_argument('mode_types', type=str, nargs='*',
                    help='mode types for which controllers will be trained')
    args = ap.parse_args()

    automaton = make_pick_place_model(num_objects=args.num_objects)
    mode_type_list = [mt.name for mt in ModeType] if args.all else args.mode_types
    for mt in mode_type_list:
        print(f'training mode type {mt}')
        names = [f'{mt}_{i}' for i in range(args.num_objects)]
        train_single(
            automaton, names, args.timesteps,
            args.path, mt,
        )
