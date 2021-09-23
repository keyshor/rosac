import os
import sys
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from stable_baselines.td3.policies import FeedForwardPolicy
from hybrid_gym.train.single_mode import (
    train_sb3, make_sb3_model, make_ars_model, learn_ars_model)
from hybrid_gym.train.mode_pred import train_mode_predictor
from hybrid_gym.hybrid_env import HybridEnv
from hybrid_gym.selectors import UniformSelector, MaxJumpWrapper
from hybrid_gym.util.wrappers import Sb3CtrlWrapper
from hybrid_gym.util.io import parse_command_line_options
from hybrid_gym.util.test import get_rollout
from hybrid_gym.envs import make_rooms_model
from hybrid_gym.rl.ars import ARSParams, NNParams, NNPolicy


def train_single(automaton, name, total_timesteps, save_path):
    mode = automaton.modes[name]
    mode_info = [(mode, automaton.transitions[name], None, None)]
    model = make_sb3_model(
        mode_info,
        algo_name='td3',
        batch_size=256,
        policy_kwargs={'net_arch': [32, 32]},
        action_noise_scale=0.15,
        learning_rate=0.0003,
        tau=0.001,
        buffer_size=50000,
        verbose=0,
        max_episode_steps=25,
        device='cpu'
    )
    train_sb3(model, mode_info,
              total_timesteps=total_timesteps, algo_name='td3',
              max_episode_steps=25,  # n_eval_episodes=10,
              save_path=save_path)


def train_ars(automaton, name, total_episodes, save_path, use_gpu):
    mode = automaton.modes[name]
    mode_info = [(mode, automaton.transitions[name], None, None)]
    nn_params = NNParams(2, 2, 1.0, 32)
    ars_params = ARSParams(total_episodes, 30, 15, 0.05, 0.3, 0.95, 25)
    model = make_ars_model(ars_params, nn_params, use_gpu)
    learn_ars_model(model, mode_info, save_path, name, True)


if __name__ == '__main__':

    flags = parse_command_line_options()
    automaton = make_rooms_model()
    mode_list = list(automaton.modes)

    if not os.path.exists(flags['path']):
        os.makedirs(flags['path'])

    for name in mode_list:
        print(f'training mode {name}')
        train_ars(automaton, name, 20000, flags['path'], flags['gpu'])

        if flags['render']:
            print('Rendering learned controller for mode {}'.format(name))
            controller = NNPolicy.load(name, flags['path'])
            for i in range(10):
                print('\n----- Rollout #{} -----'.format(i))
                get_rollout(automaton.modes[name], automaton.transitions[name], controller,
                            max_timesteps=25, render=True)
