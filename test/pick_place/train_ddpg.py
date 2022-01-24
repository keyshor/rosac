import os
import sys
import argparse
import pathlib
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa
import numpy as np
import gym
import joblib

from hybrid_gym.train.single_mode import (
        make_ddpg_model, learn_ddpg_model, env_from_mode_info,
)
from hybrid_gym.rl.ddpg import DDPGParams
from hybrid_gym.envs import make_pick_place_model
from hybrid_gym.envs.pick_place.mode import ModeType

max_episode_steps: int = 20

def train_single(automaton, names, num_episodes, save_path, mt):
    mode_info = [(automaton.modes[n], automaton.transitions[n], None, None)
                 for n in names]
    env = env_from_mode_info(
            raw_mode_info=mode_info,
            algo_name='ddpg',
            max_episode_steps=max_episode_steps,
            reward_offset=0.0,
            is_goal_env=False,
    )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high
    params = DDPGParams(
            state_dim, action_dim, action_bound,
            minibatch_size=256, num_episodes=num_episodes,
            discount=0.95, actor_hidden_dim=256,
            critic_hidden_dim=256, epsilon_decay=3e-6,
            decay_function='linear', steps_per_update=100,
            gradients_per_update=100, buffer_size=200000,
            sigma=0.15, epsilon_min=0.3, target_noise=0.0003,
            target_clip=0.003, warmup=1000,
            max_timesteps=max_episode_steps,
            test_freq=1000,
            test_n_rollouts=100,
    )
    model = make_ddpg_model(params, use_gpu=True)
    learn_ddpg_model(model, mode_info)
    joblib.dump(model, os.path.join(save_path, mt + '.ddpg'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='train controllers and mode predictor')
    ap.add_argument('--path', type=pathlib.Path, default='.',
                    help='directory in which models will be saved')
    ap.add_argument('--num-objects', type=int, default=3,
                    help='number of objects')
    ap.add_argument('--num-episodes', type=int, default=10000,
                    help='number of episodes to train each controller')
    ap.add_argument('--all', action='store_true',
                    help='use this flag to train all modes instead of specifying a list')
    ap.add_argument('mode_types', type=str, nargs='*',
                    help='mode types for which controllers will be trained')
    args = ap.parse_args()

    args.path.mkdir(parents=True, exist_ok=True)
    automaton = make_pick_place_model(num_objects=args.num_objects, reward_type='dense')
    mode_type_list = [mt.name for mt in ModeType] if args.all else args.mode_types
    for mt in mode_type_list:
        print(f'training mode type {mt}')
        names = [f'{mt}_{i}' for i in range(args.num_objects)]
        train_single(
            automaton, names, args.num_episodes,
            args.path, mt,
        )
