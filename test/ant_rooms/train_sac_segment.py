import os
import sys
import argparse
import pathlib
import pickle
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from hybrid_gym.train.single_mode import (
        make_sac_model, parallel_pool_sac, ParallelPoolArg,
)
from hybrid_gym.train.cegrl_mypool import ResetFunc
from hybrid_gym.train.mode_pred import train_mode_predictor
from hybrid_gym.hybrid_env import HybridEnv
from hybrid_gym.selectors import UniformSelector, MaxJumpWrapper
from hybrid_gym.util.wrappers import BaselineCtrlWrapper
from hybrid_gym.util.io import parse_command_line_options
from hybrid_gym.envs.ant_rooms.hybrid_env import make_ant_model


def group_index(mode_name):
    if mode_name == 'STRAIGHT':
        return 0
    elif mode_name == 'LEFT':
        return 1
    elif mode_name == 'RIGHT':
        return 2
    else:
        raise ValueError

def train_single(automaton, name,
                 steps_per_epoch,
                 num_epochs,
                 num_iter,
                 save_path,
                 use_reset_func,
                 verbose,
                 ) -> None:
    mode = automaton.modes[name]
    model = make_sac_model(
        obs_space=mode.observation_space, act_space=mode.action_space,
        hidden_dims=(256, 256),
        steps_per_epoch=steps_per_epoch, epochs=num_epochs,
        replay_size=1000000,
        gamma=1 - 1e-2, polyak=1 - 5e-3, lr=3e-4,
        alpha=0.1,
        batch_size=256,
        start_steps=10000, update_after=10000,
        update_every=50,
        num_test_episodes=30,
        max_ep_len=500, test_ep_len=500,
        log_interval=200,
        min_alpha=0.1,
        alpha_decay=1e-2,
    )

    g = group_index(name)
    tmp_filename = os.path.join(save_path, 'tmp', f'{g}_0.pkl')
    rf = ResetFunc(mode=mode) if use_reset_func else None

    model.cpu()
    with open(tmp_filename, 'wb') as f:
        pickle.dump(model, f)
    for i in range(num_iter):
        if rf:
            rf.make_serializable()
        arg = ParallelPoolArg(
            g=g, e=0, env_name='ant_rooms',
            mode_info=[(name, rf, None)],
            save_path=os.path.join(save_path, 'tmp'),
            verbose=verbose,
            retrain=(i > 0),
            use_gpu=True,
        )
        parallel_pool_sac(arg)
        if rf:
            rf.recover_after_serialization(automaton)
    with open(tmp_filename, 'rb') as f:
        model = pickle.load(f)
    policy = model.get_policy()
    policy.save(name=f'{name}_final', path=save_path)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='train controllers and mode predictor')
    ap.add_argument('--path', type=pathlib.Path, default='.',
                    help='directory in which models will be saved')
    ap.add_argument('--steps-per-epoch', type=int, default=100000,
                    help='number of training episodes per epoch')
    ap.add_argument('--num-epochs', type=int, default=3,
                    help='number of epochs to train each controller')
    ap.add_argument('--num-iter', type=int, default=2,
                    help='number of iterations to train each controller')
    ap.add_argument('--use-reset-func', action='store_true',
                    help='use this flag to train with reset funcs')
    ap.add_argument('--all', action='store_true',
                    help='use this flag to train all modes instead of specifying a list')
    ap.add_argument('--verbose', action='store_true',
                    help='controls verbosity during training')
    ap.add_argument('modes', type=str, nargs='*',
                    help='modes for which controllers will be trained')
    args = ap.parse_args()

    automaton = make_ant_model(success_bonus=1e2)
    mode_list = list(automaton.modes.keys()) if args.all else args.modes
    os.makedirs(os.path.join(args.path, 'tmp'), exist_ok=True)
    for name in mode_list:
        print(f'training mode {name}')
        train_single(
            automaton, name,
            args.steps_per_epoch, args.num_epochs, args.num_iter,
            args.path, args.use_reset_func, args.verbose,
        )
