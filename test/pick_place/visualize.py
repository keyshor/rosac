import os
import sys
import gym
import numpy as np
from stable_baselines.her import HERGoalEnvWrapper
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.util.wrappers import BaselineCtrlWrapper, SpectrlCtrlWrapper, GymEnvWrapper, GymGoalEnvWrapper
from hybrid_gym.envs import make_pick_place_model
from hybrid_gym.envs.pick_place.mode import ModeType
from hybrid_gym.hybrid_env import HybridEnv, HybridGoalEnv
from hybrid_gym.selectors import UniformSelector

num_objects = 3
automaton = make_pick_place_model(num_objects=num_objects,
                                  reward_type='sparse',
                                  distance_threshold=0.015)

# env = HybridEnv(
env = HybridGoalEnv(
    automaton=automaton,
    selector=UniformSelector(modes=automaton.modes.values()),
    # flatten_obs=True
)
controllers: dict = {}

num_trials = 5
steps_per_trial = 100


def visualize_mode(name):
    mode = automaton.modes[name]
    #mode_env = GymEnvWrapper(mode, automaton.transitions[name], flatten_obs=True)
    mode_env = GymGoalEnvWrapper(mode, automaton.transitions[name])
    ctrl = controllers[name]
    num_successes = 0
    for _ in range(num_trials):
        obs = mode_env.reset()
        done = False
        for _ in range(steps_per_trial):
            # if not done:
            #    action = ctrl.get_action(obs)
            #    obs, _, done, _ = mode_env.step(action)
            mode_env.render()
        if mode.is_success(mode_env.state):
            num_successes += 1
    #print(f'success rate for mode {name} is {num_successes}/{num_trials}')


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        mode_list = sys.argv[1:]
    else:
        mode_list = list(automaton.modes)
    for mt in ModeType:
        ctrl = BaselineCtrlWrapper.load(
            os.path.join('her_models', mt.name, 'best_model.zip'),
            algo_name='her',
            env=env,
        )
        for i in range(num_objects):
            controllers[f'{mt.name}_{i}'] = ctrl

    for mt in ModeType:
        i = np.random.choice(num_objects)
        mode_name = f'{mt.name}_{i}'
        print(f'visualizing {mode_name}')
        visualize_mode(mode_name)
