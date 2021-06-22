import os
import sys
import numpy as np
import gym
from stable_baselines import SAC, HER
from stable_baselines.ddpg.noise import NormalActionNoise
from stable_baselines.common.vec_env import DummyVecEnv
from gym.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
from gym.wrappers import TimeLimit
sys.path.append(os.path.join('..', '..'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.train.single_mode import make_sb_model, train_stable
from hybrid_gym.util.wrappers import BaselineCtrlWrapper, DoneOnSuccessWrapper
from hybrid_gym.train.mode_pred import train_mode_predictor
from hybrid_gym.envs import make_pick_place_model
from hybrid_gym.envs.pick_place.mode import PickPlaceMode

if __name__ == '__main__':
    automaton = make_pick_place_model(num_objects=3)
    # models = {
    #    name: make_sb_model(
    #        mode,
    #        automaton.transitions[name],
    #        algo_name='her',
    #        wrapped_algo='sac',
    #        gamma=0.95, buffer_size=1000000,
    #        ent_coef='auto', goal_selection_strategy='future',
    #        n_sampled_goal=4, train_freq=1, learning_starts=1000,
    #        verbose=2
    #    )
    #    for (name, mode) in automaton.modes.items()
    # }
    # for (name, mode) in [
    #        ('ModeType.PLACE_OBJ_PT2', automaton.modes['ModeType.PLACE_OBJ_PT2']),
    # ]:
    for (name, mode) in automaton.modes.items():
        model = make_sb_model(
            mode,
            automaton.transitions[name],
            algo_name='her',
            wrapped_algo='sac',
            gamma=0.95, buffer_size=1000000,
            ent_coef='auto', goal_selection_strategy='future',
            n_sampled_goal=4, train_freq=1, learning_starts=1000,
            verbose=2
        )
        train_stable(model, mode, automaton.transitions[name],
                     total_timesteps=500000, algo_name='her')
        ctrl = BaselineCtrlWrapper(model)
        ctrl.save(f'{name}.her')
    #controller = {name: BaselineCtrlWrapper(model) for (name, model) in models.items()}
    # for (mode_name, ctrl) in [('ModeType.PLACE_OBJ', controller['ModeType.PLACE_OBJ'])]:
    # for (mode_name, ctrl) in controller.items():
    #    ctrl.save(f'{mode_name}.her')
    #env = DoneOnSuccessWrapper(TimeLimit(FetchPickAndPlaceEnv(), max_episode_steps=50))
    # model = HER('MlpPolicy', env, SAC, gamma=0.95, buffer_size=1000000,
    #            ent_coef='auto', goal_selection_strategy='future',
    #            n_sampled_goal=4, train_freq=1, learning_starts=1000, verbose=2)
    # model.learn(total_timesteps=200000)
    # model.save('pick_place_test.her')
