import os
import sys
import numpy as np
import gym
import argparse
import pathlib
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from gym.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
from gym.wrappers import TimeLimit
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.train.single_mode import train_sb3, make_sb3_model_init_check, make_sb3_model
from hybrid_gym.train.mode_pred import train_mode_predictor
from hybrid_gym.envs import make_pick_place_model
from hybrid_gym.envs.pick_place.mode import PickPlaceMode, ModeType
from hybrid_gym.util.wrappers import DoneOnSuccessWrapper


max_episode_steps=20

def train_single(automaton, name, total_timesteps, save_path, model_path):
    mode_info = [(automaton.modes[name], automaton.transitions[name], None, None)]
    model = make_sb3_model(
        mode_info,
        algo_name='td3',
        policy='MlpPolicy',
        buffer_size=200000,
        action_noise_scale=0.15,
        batch_size=256,
        gamma=0.95,
        learning_rate=1e-4,
        policy_kwargs=dict(
            net_arch=[1024, 1024, 1024],
        ),
        target_policy_noise=3e-4,
        target_noise_clip=3e-3,
        reward_offset=0.0,
        max_episode_steps=max_episode_steps,
        is_goal_env=False,
        verbose=0,
    )
    train_sb3(model, mode_info,
              total_timesteps=total_timesteps, algo_name='td3',
              reward_offset=0.0, is_goal_env=False, use_best_model=True,
              max_episode_steps=max_episode_steps,
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

    automaton = make_pick_place_model(num_objects=args.num_objects, reward_type='dense')
    mode_type_list = [mt.name for mt in ModeType] if args.all else args.mode_types
    for mt in mode_type_list:
        print(f'training mode type {mt}')
        for i in range(args.num_objects):
            name = f'{mt}_{i}'
            print(f'training mode {name}')
            train_single(
                automaton, name, args.timesteps,
                args.path, name,
            )
