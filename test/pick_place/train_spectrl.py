import os
import sys
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa
import numpy as np
import gym
import joblib

from hybrid_gym.train.single_mode import make_spectrl_model, train_spectrl
from hybrid_gym.envs import make_pick_place_model

automaton = make_pick_place_model(num_objects=3, reward_type='dense')


def train_single(name, num_episodes, save_path):
    mode = automaton.modes[name]
    model = make_spectrl_model(
        mode,
        automaton.transitions[name],
        minibatch_size=256, num_episodes=num_episodes,
        discount=0.95, actor_hidden_dim=256,
        critic_hidden_dim=256, epsilon_decay=3e-6,
        decay_function='linear', steps_per_update=100,
        gradients_per_update=100, buffer_size=200000,
        sigma=0.15, epsilon_min=0.3, target_noise=0.0003,
        target_clip=0.003, warmup=1000,
    )
    train_spectrl(model, mode, automaton.transitions[name])
    joblib.dump(model, os.path.join(save_path, mode.name + '.spectrl'))


if __name__ == '__main__':
    train_single(sys.argv[2], int(sys.argv[3]), sys.argv[1])
    # for name in automaton.modes:
    #    train_single(name, 4000)
