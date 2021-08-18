import os
import sys
import argparse
import pathlib
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
from hybrid_gym.envs.pick_place.mode import PickPlaceMode, ModeType


def train_single(automaton, names, total_timesteps, save_path, model_path):
    modes = [automaton.modes[n] for n in names]
    mode_info = [(automaton.modes[n], automaton.transitions[n], None, None)
                 for n in names]
    model = make_sb_model(
        modes,
        algo_name='her',
        wrapped_algo='sac',
        gamma=0.95, buffer_size=1000000,
        ent_coef='auto', goal_selection_strategy='future',
        n_sampled_goal=4, train_freq=1, learning_starts=1000,
        verbose=2
    )
    train_stable(model, mode_info,
                 total_timesteps=total_timesteps, algo_name='her',
                 save_path=save_path, custom_best_model_path=model_path)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='train controllers and mode predictor')
    ap.add_argument('--path', type=pathlib.Path, default='.',
                    help='directory in which models will be saved')
    ap.add_argument('--num-objects', type=int, default=3,
                    help='number of objects')
    ap.add_argument('--timesteps', type=int, default=50000,
                    help='number of timesteps to train each controller')
    ap.add_argument('--all', action='store_true',
                    help='use this flag to train all modes instead of specifying a list')
    ap.add_argument('mode-types', type=str, nargs='*',
                    help='mode types for which controllers will be trained')
    args = ap.parse_args()

    automaton = make_pick_place_model(num_objects=args.num_objects)
    mode_type_list = [mt.name for mt in ModeType] if args.all else args.mode_types
    for mt in mode_type_list:
        names = [f'{mt}_{i}' for i in range(args.num_objects)]
        train_single(
            automaton, names, args.timesteps,
            args.path, mt,
        )
