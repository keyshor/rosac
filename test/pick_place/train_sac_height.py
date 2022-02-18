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
from hybrid_gym.train.single_mode import make_sac_model, learn_sac_model
from hybrid_gym.train.mode_pred import train_mode_predictor
from hybrid_gym.envs import make_pick_place_model
from hybrid_gym.envs.pick_place.mode import PickPlaceMode, ModeType
from hybrid_gym.util.wrappers import DoneOnSuccessWrapper

max_episode_steps=20


def train_single(automaton, names,
                 episodes_per_epoch, num_epochs,
                 save_path, model_path, verbose):
    modes = [automaton.modes[n] for n in names]
    mode_info = [(automaton.modes[n], automaton.transitions[n], None, None)
                 for n in names]
    model = make_sac_model(
        obs_space=modes[0].observation_space, act_space=modes[0].action_space,
        hidden_dims=(1024, 1024, 1024),
        episodes_per_epoch=episodes_per_epoch, epochs=num_epochs,
        replay_size=1000000,
        gamma=1 - 5e-2, polyak=1 - 5e-3, lr=1e-3,
        alpha=0.1,
        batch_size=256,
        start_steps=10000, update_after=1000,
        update_every=50,
        num_test_episodes=100,
        max_ep_len=max_episode_steps, test_ep_len=max_episode_steps,
        log_interval=episodes_per_epoch,
        min_alpha=0.1,
        alpha_decay=1e-2,
    )
    learn_sac_model(
        model=model,
        raw_mode_info=mode_info,
        verbose=verbose,
        retrain=False,
    )
    policy = model.get_policy()
    policy.save(name=model_path, path=save_path)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='train controllers and mode predictor')
    ap.add_argument('--path', type=pathlib.Path, default='.',
                    help='directory in which models will be saved')
    ap.add_argument('--num-objects', type=int, default=3,
                    help='number of objects')
    ap.add_argument('--episodes-per-epoch', type=int, default=50,
                    help='number of training episodes per epoch')
    ap.add_argument('--num-epochs', type=int, default=100,
                    help='number of epochs to train each controller')
    ap.add_argument('--all', action='store_true',
                    help='use this flag to train all modes instead of specifying a list')
    ap.add_argument('--verbose', action='store_true',
                    help='use this flag to print additional information during training')
    ap.add_argument('mode_types', type=str, nargs='*',
                    help='mode types for which controllers will be trained')
    args = ap.parse_args()

    args.path.mkdir(parents=True, exist_ok=True)
    automaton = make_pick_place_model(
        num_objects=args.num_objects, reward_type='dense',
        fixed_tower_height=True, flatten_obs=True,
    )
    mode_type_list = [mt.name for mt in ModeType] if args.all else args.mode_types
    for mt in mode_type_list:
        print(f'training mode type {mt}')
        names = [f'{mt}_{i}' for i in range(args.num_objects)]
        if mt == 'MOVE_WITH_OBJ':
            #for j in range(args.num_objects):
            for j in [1]:
                #names = [f'{mt}_h{j}_{i}' for i in range(args.num_objects)]
                #print(f'training height {j}')
                #train_single(
                #    automaton, names, args.timesteps,
                #    args.path, f'{mt}_h{j}',
                #)
                for i in range(args.num_objects):
                    name = f'{mt}_h{j}_{i}'
                    print(f'training mode {name}')
                    train_single(
                        automaton, [name],
                        args.episodes_per_epoch, args.num_epochs,
                        args.path, name, args.verbose,
                    )
        else:
            train_single(
                automaton, names,
                args.episodes_per_epoch, args.num_epochs,
                args.path, mt, args.verbose
            )
