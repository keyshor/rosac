import os
import sys
import argparse
import pathlib
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.train.single_mode import train_stable, make_sb_model
from hybrid_gym.train.mode_pred import train_mode_predictor
from hybrid_gym.hybrid_env import HybridEnv
from hybrid_gym.selectors import UniformSelector, MaxJumpWrapper
from hybrid_gym.util.wrappers import BaselineCtrlWrapper
from hybrid_gym.envs.rooms.hybrid_env import make_rooms_model

def train_single(automaton, name, total_timesteps, save_path):
    mode = automaton.modes[name]
    model = make_sb_model(
        mode,
        automaton.transitions[name],
        algo_name='td3',
    )
    train_stable(model, mode, automaton.transitions[name],
                 total_timesteps=total_timesteps, algo_name='td3',
                 max_episode_steps=100, save_path=save_path,
                 eval_freq=1000,
                 )

def train_mode_pred(automaton, save_path, num_iters, samples_per_mode_per_iter):
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
        activation='tanh', samples_per_mode_per_iter=samples_per_mode_per_iter,
    )
    mode_pred.save(os.path.join(save_path, 'mode_predictor.mlp'))

if __name__ == '__main__':
    automaton = make_rooms_model()
    ap = argparse.ArgumentParser(description='train controllers and mode predictor')
    ap.add_argument('--path', type=pathlib.Path, default='.',
                    help='directory in which models will be saved')
    ap.add_argument('--timesteps', type=int, default=3000,
                    help='number of timesteps to train each controller')
    ap.add_argument('--all', action='store_true',
                    help='use this flag to train all modes instead of specifying a list')
    ap.add_argument('--mode_pred', action='store_true',
                    help='use this flag to train a mode predictor')
    ap.add_argument('--pred_iters', type=int, default=10,
                    help='number of iterations for mode predictor training')
    ap.add_argument('--pred_samples', type=int, default=200,
                    help='samples per mode for mode predictor training')
    ap.add_argument('modes', type=str, nargs='*',
                    help='modes for which controllers will be trained')
    args = ap.parse_args()

    mode_list = list(automaton.modes) if args.all else args.modes
    args.path.mkdir(parents=True, exist_ok=True)
    for name in mode_list:
        print(f'training mode {name}')
        train_single(automaton, name, args.timesteps, args.path)
    if args.mode_pred:
        train_mode_pred(automaton, args.path, args.pred_iters, args.pred_samples)
