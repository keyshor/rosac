import os
import sys
sys.path.append(os.path.join('..', '..'))  # nopep8

from hybrid_gym.train.single_mode import BaselineCtrlWrapper
from hybrid_gym.envs import make_f110_model
from hybrid_gym.hybrid_env import HybridEnv
from hybrid_gym.selectors import UniformSelector, MaxJumpWrapper
from typing import Dict, List

import matplotlib.pyplot as plt


if __name__ == '__main__':
    automaton = make_f110_model(straight_lengths=[10])
    controllers = {name: BaselineCtrlWrapper.load(f'{name}.td3', algo_name='td3')
                   for name in automaton.modes}

    test_env = HybridEnv(
        automaton=automaton,
        selector=MaxJumpWrapper(
            wrapped_selector=UniformSelector(modes=automaton.modes.values()),
            max_jumps=10
        )
    )

    state_history: Dict[str, List] = {m_name: [] for m_name in automaton.modes}

    observation = test_env.reset()

    e = 0
    done = False
    while not done:
        e += 1
        # observation = normalize(observation)
        delta = controllers[test_env.mode.name].get_action(observation)
        state_history[test_env.mode.name].append(test_env.state)
        observation, reward, done, info = test_env.step(delta)
        if not test_env.mode.is_safe(test_env.state):
            print('Crash after {} steps'.format(e))

    for (m_name, m) in automaton.modes.items():
        # fig = plt.figure(figsize=(12, 10))
        fig = plt.figure()
        m.plotHalls()
        x_hist = [s.car_global_x for s in state_history[m_name]]
        y_hist = [s.car_global_y for s in state_history[m_name]]
        plt.scatter(x_hist, y_hist, s=1, c='r')
        plt.show()
        plt.savefig(f'trajectories_{m.name}.png')
