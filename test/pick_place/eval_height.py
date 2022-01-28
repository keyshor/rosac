import os
import argparse
import pathlib
import sys
import gym
import numpy as np
from stable_baselines.her import HERGoalEnvWrapper
import torch
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.util.wrappers import Sb3CtrlWrapper, SpectrlCtrlWrapper, DdpgCtrlWrapper, GymEnvWrapper, GymGoalEnvWrapper
from hybrid_gym.envs import make_pick_place_model
from hybrid_gym.envs.pick_place.mode import ModeType
from hybrid_gym.hybrid_env import HybridEnv, HybridGoalEnv
from hybrid_gym.selectors import UniformSelector



def eval_single(automaton, controllers, name, trials_per_mode, steps_per_trial, render, print_reward):
    mode = automaton.modes[name]
    mode_env = GymEnvWrapper(mode, automaton.transitions[name], flatten_obs=True)
    #mode_env = GymGoalEnvWrapper(mode, automaton.transitions[name])
    ctrl = controllers[name]
    num_successes = 0
    min_success_time = steps_per_trial + 1
    max_success_time = -1
    for _ in range(trials_per_mode):
        obs = mode_env.reset()
        done = False
        num_steps = 0
        for j in range(steps_per_trial):
            if not done:
                action = ctrl.get_action(obs)
                obs, reward, done, _ = mode_env.step(action)
                num_steps += 1
            if render:
                mode_env.render()
            if print_reward:
                print(f'reward at step {j} is {reward}')
        if mode.is_success(mode_env.state):
            num_successes += 1
            min_success_time = min(min_success_time, num_steps)
            max_success_time = max(max_success_time, num_steps)
    print(f'{name}: success {num_successes}/{trials_per_mode}, success time [{min_success_time}, {max_success_time}]')


def eval_end_to_end(automaton, controllers, end_to_end_trials, steps_per_mode):
    #env = HybridGoalEnv(
    env = HybridEnv(
        automaton=automaton,
        selector=UniformSelector(modes=[
            automaton.modes[f'MOVE_WITHOUT_OBJ_{i}']
            for i in range(args.num_objects)
        ]),
        flatten_obs=True,
    )
    num_successes = 0
    for _ in range(end_to_end_trials):
        observation = env.reset()
        steps_in_cur_mode = 0
        mode = ''
        done = False
        while not done:
            if mode != env.mode.name:
                mode = env.mode.name
                steps_in_cur_mode = 0
                #print(f'switched to {mode}')
            delta = controllers[mode].get_action(observation)
            observation, reward, done, info = env.step(delta)
            # env.render()
            steps_in_cur_mode += 1
            if steps_in_cur_mode > steps_per_mode:
                #print(f'stuck in mode {mode} for {steps_per_mode} steps')
                break
        if all([i in env.mode.multi_obj.tower_set for i in range(env.mode.multi_obj.num_objects)]):
            num_successes += 1
    print(f'end-to-end success rate is {num_successes}/{end_to_end_trials}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='evaluate controllers and mode predictor')
    ap.add_argument('--path', type=pathlib.Path, default='.',
                    help='directory in which models are stored')
    ap.add_argument('--num-objects', type=int, default=3,
                    help='number of objects')
    ap.add_argument('--goal-tolerance', type=float, default=0.015,
                    help='maximum distance to goal that is treated as a success')
    ap.add_argument('--trials-per-mode', type=int, default=100,
                    help='number of trials for each mode in single-mode evaluation')
    ap.add_argument('--steps-per-trial', type=int, default=100,
                    help='maximum number of steps per trial in single-mode evaluation')
    ap.add_argument('--end-to-end-trials', type=int, default=0,
                    help='number of end-to-end trials')
    ap.add_argument('--steps-per-mode', type=int, default=100,
                    help='maximum number of steps per mode in end-to-end evaluation')
    ap.add_argument('--render', action='store_true',
                    help='use this flag to render the environment at each time-step')
    ap.add_argument('--print-reward', action='store_true',
                    help='use this flag to print the reward at each time-step')
    ap.add_argument('--all', action='store_true',
                    help='use this flag to train all modes instead of specifying a list')
    #ap.add_argument('mode_types', type=str, nargs='*',
    #                help='mode types for which controllers will be evaluated')
    ap.add_argument('modes', type=str, nargs='*',
                    help='modes for which controllers will be evaluated')
    args = ap.parse_args()
    automaton = make_pick_place_model(
        num_objects=args.num_objects,
        reward_type='dense',
        distance_threshold=args.goal_tolerance,
        fixed_tower_height=True,
    )
    #mode_type_list = [mt.name for mt in ModeType] if args.all else args.mode_types
    mode_list = [name for name in automaton.modes.keys()
                 if 'MOVE_WITH_OBJ_' in name] \
        if args.all else args.modes

    #env = HybridGoalEnv(
    env = HybridEnv(
        automaton=automaton,
        selector=UniformSelector(modes=[
            automaton.modes[f'MOVE_WITHOUT_OBJ_{i}']
            for i in range(args.num_objects)
        ]),
        flatten_obs=True,
    )
    #controllers: dict = {}
    #for mt_name in mode_type_list:
    #for mt in ModeType:
    #    if mt.name == 'MOVE_WITH_OBJ':
    #        for j in range(args.num_objects):
    #            #name = f'{mt.name}_h{j}'
    #            #ctrl = Sb3CtrlWrapper.load(
    #            #    os.path.join(args.path, name, 'best_model.zip'),
    #            #    algo_name='td3',
    #            #    env=env,
    #            #)
    #            for i in range(args.num_objects):
    #                name = f'{mt.name}_h{j}_{i}'
    #                controllers[name] = Sb3CtrlWrapper.load(
    #                    os.path.join(args.path, name, 'best_model.zip'),
    #                    algo_name='td3',
    #                    env=env,
    #                )
    #    #else:
    #    #    ctrl = Sb3CtrlWrapper.load(
    #    #        os.path.join(args.path, mt.name, 'best_model.zip'),
    #    #        algo_name='td3',
    #    #        env=env,
    #    #    )
    #    #    for i in range(args.num_objects):
    #    #        controllers[f'{mt.name}_{i}'] = ctrl
    controllers = {
        name: Sb3CtrlWrapper.load(
            os.path.join(args.path, name, 'best_model.zip'),
            algo_name='td3',
            env=env,
        )
        for name in mode_list
    }
    for mode in mode_list:
        #print(f'evaluating mode {mode}')
        eval_single(automaton, controllers, mode, args.trials_per_mode, args.steps_per_trial, args.render, args.print_reward)
    eval_end_to_end(automaton, controllers, args.end_to_end_trials, args.steps_per_mode)
