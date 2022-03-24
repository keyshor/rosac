import os
import sys
import argparse
import pathlib
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from hybrid_gym.train.single_mode import make_sac_model, learn_sac_model
from hybrid_gym.train.mode_pred import train_mode_predictor
from hybrid_gym.hybrid_env import HybridEnv
from hybrid_gym.selectors import UniformSelector, MaxJumpWrapper
from hybrid_gym.util.wrappers import BaselineCtrlWrapper
from hybrid_gym.util.io import parse_command_line_options
from hybrid_gym.envs.f110_turn.hybrid_env import make_f110_model


def train_single(automaton, name,
                 steps_per_epoch,
                 num_epochs,
                 save_path,
                 verbose,
                 ) -> None:
    mode = automaton.modes[name]
    mode_info = [(mode, automaton.transitions[name], None, None)]
    init_ok = False
    model = make_sac_model(
        obs_space=mode.observation_space, act_space=mode.action_space,
        hidden_dims=(64, 64),
        steps_per_epoch=steps_per_epoch, epochs=num_epochs,
        replay_size=50000,
        gamma=1 - 1e-2, polyak=1 - 5e-3, lr=3e-4,
        alpha=0.1,
        batch_size=256,
        start_steps=2000, update_after=2000,
        update_every=50,
        num_test_episodes=10,
        max_ep_len=50, test_ep_len=50,
        log_interval=100,
        min_alpha=0.1,
        alpha_decay=1e-2,
    )
    learn_sac_model(
        automaton=automaton,
        model=model,
        raw_mode_info=mode_info,
        verbose=verbose,
        retrain=False,
        print_success_rate=True,
    )
    policy = model.get_policy()
    policy.save(name=name, path=save_path)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='train controllers and mode predictor')
    ap.add_argument('--path', type=pathlib.Path, default='.',
                    help='directory in which models will be saved')
    ap.add_argument('--steps-per-epoch', type=int, default=10000,
                    help='number of training episodes per epoch')
    ap.add_argument('--num-epochs', type=int, default=2,
                    help='number of epochs to train each controller')
    ap.add_argument('--all', action='store_true',
                    help='use this flag to train all modes instead of specifying a list')
    ap.add_argument('--verbose', action='store_true',
                    help='controls verbosity during training')
    ap.add_argument('modes', type=str, nargs='*',
                    help='modes for which controllers will be trained')
    args = ap.parse_args()

    automaton = make_f110_model()
    mode_list = list(automaton.modes.keys()) if args.all else args.modes
    args.path.mkdir(parents=True, exist_ok=True)
    for name in mode_list:
        print(f'training mode {name}')
        train_single(
            automaton, name,
            args.steps_per_epoch, args.num_epochs,
            args.path, args.verbose,
        )
