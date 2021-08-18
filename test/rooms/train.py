import os
import sys
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from stable_baselines.td3.policies import FeedForwardPolicy
from hybrid_gym.train.single_mode import train_stable, make_sb_model
from hybrid_gym.train.mode_pred import train_mode_predictor
from hybrid_gym.hybrid_env import HybridEnv
from hybrid_gym.selectors import UniformSelector, MaxJumpWrapper
from hybrid_gym.util.wrappers import BaselineCtrlWrapper
from hybrid_gym.util.io import parse_command_line_options
from hybrid_gym.envs import make_rooms_model


def train_single(automaton, name, total_timesteps, save_path):
    mode = automaton.modes[name]
    model = make_sb_model(
        mode,
        automaton.transitions[name],
        algo_name='td3',
        batch_size=256,
        policy_kwargs={'layers': [32, 32]},
        action_noise_scale=0.15,
        verbose=0,
        max_episode_steps=25,
    )
    train_stable(model, mode, automaton.transitions[name],
                 total_timesteps=total_timesteps, algo_name='td3',
                 max_episode_steps=25, eval_freq=1000,
                 n_eval_episodes=10, save_path=save_path)


if __name__ == '__main__':

    flags = parse_command_line_options()
    automaton = make_rooms_model()
    mode_list = list(automaton.modes)

    if not os.path.exists(flags['path']):
        os.makedirs(flags['path'])

    for name in mode_list:
        print(f'training mode {name}')
        train_single(automaton, name, 400000, flags['path'])
