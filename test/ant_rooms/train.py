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

from hybrid_gym.train.single_mode import train_sb3, make_sb3_model_init_check, make_sb3_model
from hybrid_gym.train.mode_pred import train_mode_predictor
from hybrid_gym.hybrid_env import HybridEnv
from hybrid_gym.selectors import UniformSelector, MaxJumpWrapper
from hybrid_gym.util.wrappers import BaselineCtrlWrapper
from hybrid_gym.util.io import parse_command_line_options
from hybrid_gym.envs.ant_rooms.hybrid_env import make_ant_model


def train_single(automaton, name,
                 total_timesteps, max_init_retries,
                 net_arch,
                 save_path,
                 verbose,
                 ) -> None:
    mode = automaton.modes[name]
    mode_info = [(mode, automaton.transitions[name], None, None)]
    init_ok = False
    model = make_sb3_model(
        mode_info,
        algo_name='td3',
        policy_kwargs=dict(
            net_arch=net_arch,
        ),
        action_noise_scale=0.1,
        verbose=verbose,
        #max_episode_steps=100,
        #max_init_retries=10,
        #min_reward=-np.inf,
        #min_episode_length=10,
    )
    train_sb3(model, mode_info,
              total_timesteps=total_timesteps, algo_name='td3',
              max_episode_steps=100, save_path=save_path)

def train_monolithic(mono_automaton, total_timesteps, net_arch, save_path, verbose):
    env = HybridEnv(
        automaton=mono_automaton,
        selector=MaxJumpWrapper(
            wrapped_selector=UniformSelector(modes=automaton.modes.values()),
            max_jumps=100
        ),
        max_timesteps=1000000000000000000000000,
        max_timesteps_per_mode=100,
    )
    action_shape = env.action_space.shape
    action_noise = NormalActionNoise(
        mean=np.zeros(action_shape),
        sigma=np.full(shape=action_shape, fill_value=0.1)
    )
    model = TD3(
        'MlpPolicy', env,
        action_noise=action_noise,
        policy_kwargs=dict(
            net_arch=net_arch,
        ),
        verbose=verbose,
    )
    callback = EvalCallback(
        eval_env=Monitor(env), n_eval_episodes=100, eval_freq=10000,
        log_path=os.path.join(save_path, 'monolithic'),
        best_model_save_path=os.path.join(save_path, 'monolithic'),
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
    )

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='train controllers and mode predictor')
    ap.add_argument('--path', type=pathlib.Path, default='.',
                    help='directory in which models will be saved')
    ap.add_argument('--timesteps', type=int, default=20000,
                    help='number of timesteps to train each controller')
    ap.add_argument('--all', action='store_true',
                    help='use this flag to train all modes instead of specifying a list')
    ap.add_argument('--monolithic-timesteps', type=int, default=0,
                    help='number of timesteps to train monolithic controller')
    ap.add_argument('--max-init-retries', type=int, default=10,
                    help='number of retries to initialize model')
    ap.add_argument('--bad-start-region', action='store_true',
                    help='use this flag to give the training algorithm a bad start region')
    ap.add_argument('--verbose', '-v', action='count', default=0,
                    help='each instance of -v increases verbosity')
    ap.add_argument('modes', type=str, nargs='*',
                    help='modes for which controllers will be trained')
    args = ap.parse_args()

    automaton = make_ant_model()
    mode_list = list(automaton.modes.keys()) if args.all else args.modes
    args.path.mkdir(parents=True, exist_ok=True)
    for name in mode_list:
        print(f'training mode {name}')
        train_single(automaton, name, args.timesteps, args.max_init_retries, [64,64], args.path, args.verbose)
    if args.monolithic_timesteps > 0:
        print(f'training monolithic controller')
        train_monolithic(
            make_f110_rooms_model(use_throttle=not args.no_throttle, observe_heading=True, observe_mode_onehot=True),
            args.monolithic_timesteps, [256, 256], args.path, args.verbose,
        )
