import os
import sys
import argparse
import pathlib
import numpy as np
import gym
#from stable_baselines import SAC, HER
#from stable_baselines.ddpg.noise import NormalActionNoise
#from stable_baselines.common.vec_env import DummyVecEnv
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

max_episode_steps = 20

sb2_hyperparams = dict(
    algo_name='sac', policy='MultiInputPolicy',
    gamma=0.95, buffer_size=1000000,
    batch_size=256, learning_rate=0.001,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy='future',
        online_sampling=True,
        max_episode_length=max_episode_steps,
        ),
    policy_kwargs=dict(
        net_arch=[64, 64],
        ),
    train_freq=1, learning_starts=1000,
)

sb3_hyperparams = dict(
    algo_name='tqc', policy='MultiInputPolicy',
    gamma=0.95, buffer_size=1000000,
    batch_size=2048, learning_rate=0.001,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy='future',
        online_sampling=True,
        max_episode_length=max_episode_steps,
    ),
    policy_kwargs=dict(
        net_arch=[512, 512, 512],
        n_critics=2,
    ),
    train_freq=1, learning_starts=1000,
)

sb3_hyperparams_sac = dict(
    algo_name='sac', policy='MultiInputPolicy',
    buffer_size=1000000,
    ent_coef='auto',
    batch_size=256,
    gamma=0.95,
    learning_rate=1e-3,
    learning_starts=1000,
    #normalize=True,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        online_sampling=True,
        goal_selection_strategy='future',
        n_sampled_goal=4,
        max_episode_length=max_episode_steps,
    ),
    policy_kwargs=dict(
        net_arch=[64,64],
    ),
)


def train_single(automaton, name, total_timesteps, save_path, model_path):
    mode_info = [(automaton.modes[name], automaton.transitions[name], None, None)]
    model = make_sb3_model(
        mode_info,
        reward_offset=0.0,
        is_goal_env=True,
        verbose=0,
        max_episode_steps=max_episode_steps,
        **sb3_hyperparams_sac,
    )
    train_sb3(model, mode_info,
              total_timesteps=total_timesteps, algo_name='her',
              max_episode_steps=max_episode_steps,
              reward_offset=0.0, is_goal_env=True,
              use_best_model=True,
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
        for i in range(args.num_objects):
            name = f'{mt}_{i}'
            print(f'training mode {name}')
            train_single(
                automaton, name, args.timesteps,
                args.path, mt,
            )
