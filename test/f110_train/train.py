import os
import sys
import argparse
import pathlib
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
import numpy as np
from stable_baselines.td3.policies import FeedForwardPolicy
from hybrid_gym.train.single_mode import train_sb3, make_sb3_model_init_check, make_sb3_model
from hybrid_gym.train.mode_pred import train_mode_predictor
from hybrid_gym.hybrid_env import HybridEnv
from hybrid_gym.selectors import UniformSelector, MaxJumpWrapper
from hybrid_gym.util.wrappers import BaselineCtrlWrapper
from hybrid_gym.util.io import parse_command_line_options
from hybrid_gym.envs.f110.hybrid_env import make_f110_model


class F110Policy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         **kwargs,
                         layers=[200, 50],
                         layer_norm=False,
                         feature_extraction='mlp',
                         )



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
        batch_size=100,
        action_noise_scale=4.0,
        learning_rate=0.0003,
        buffer_size=50000,
        verbose=verbose,
        #max_episode_steps=100,
        #max_init_retries=10,
        #min_reward=-np.inf,
        #min_episode_length=10,
    )
    train_sb3(model, mode_info,
              total_timesteps=total_timesteps, algo_name='td3',
              max_episode_steps=100, save_path=save_path)

def train_mode_pred(automaton, save_path, num_iters):
    env = HybridEnv(
        automaton=automaton,
        selector=MaxJumpWrapper(
            wrapped_selector=UniformSelector(modes=automaton.modes.values()),
            max_jumps=100
        )
    )
    controller = {
        name: BaselineCtrlWrapper.load(
            os.path.join(save_path, name, 'best_model.zip'),
            algo_name='td3',
            env=env,
        )
        for name in automaton.modes
    }
    mode_pred = train_mode_predictor(
        automaton, {}, controller, 'mlp', num_iters=num_iters,
        hidden_layer_sizes=(800,200), activation='tanh',
        samples_per_mode_per_iter=1000,
    )
    mode_pred.save(os.path.join(save_path, 'mode_predictor.mlp'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='train controllers and mode predictor')
    ap.add_argument('--path', type=pathlib.Path, default='.',
                    help='directory in which models will be saved')
    ap.add_argument('--no-throttle', action='store_true',
                    help='use this flag to disable throttle in the environment')
    ap.add_argument('--timesteps', type=int, default=20000,
                    help='number of timesteps to train each controller')
    ap.add_argument('--obstacle-timesteps', type=int, default=100000,
                    help='number of timesteps to train obstacle controller')
    ap.add_argument('--all', action='store_true',
                    help='use this flag to train all modes instead of specifying a list')
    ap.add_argument('--max-init-retries', type=int, default=10,
                    help='number of retries to initialize model')
    ap.add_argument('--mode-pred', action='store_true',
                    help='use this flag to train a mode predictor')
    ap.add_argument('--pred-iters', type=int, default=10,
                    help='number of iterations for mode predictor training')
    ap.add_argument('--verbose', '-v', action='count', default=0,
                    help='each instance of -v increases verbosity')
    ap.add_argument('modes', type=str, nargs='*',
                    help='modes for which controllers will be trained')
    args = ap.parse_args()

    automaton = make_f110_model(straight_lengths=[10], use_throttle=not args.no_throttle)
    mode_list = list(automaton.modes) if args.all else args.modes
    args.path.mkdir(parents=True, exist_ok=True)
    for name in mode_list:
        print(f'training mode {name}')
        num_timesteps = args.obstacle_timesteps if 'obstacle' in name else args.timesteps
        train_single(automaton, name, num_timesteps, args.max_init_retries, [200,50], args.path, args.verbose)
    if args.mode_pred:
        train_mode_pred(automaton, args.path, args.pred_iters)
