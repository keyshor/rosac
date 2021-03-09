import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.join('..', '..'))
from hydrid_gym.hybrid_env import HybridEnv
from hybrid_gym.envs.f110.hybrid_env import make_hybrid_env
from hybrid_gym.train_single_mode.stable_baselines_wrapper import BaselineCtrlWrapper, train_stable

if __name__ == '__main__':
    env = make_hybrid_env(straight_lengths=[10])
    controllers = {name: BaselineCtrlWrapper.load(f'{name}.td3', algo='td3')
                   for name in env.automaton.modes}

    f110_automaton = env.automaton
    test_env = HybridEnv(
        automaton=f110_automaton,
        selector=MaxJumpWrapper(
            wrapped_selector=UniformSelector(modes=f110_automaton.modes.values()),
            max_jumps=100
        )
    )

    state_history = {m_name: [] for m_name in f110_automaton.modes}

    observation = env.reset()

    e = 0
    done = False
    while not done:
        e += 1
        observation = normalize(observation)
        delta = controllers[env.mode.name].get_action(observation)
        state_history[env.mode.name, mode].append(env.state)
        observation, reward, done, info = env.step(delta)
        if not env.mode.is_safe(env.state):
            print('Crash after {} steps'.format(e))

    for (m_name, m) in f110_automaton.modes.items():
        #fig = plt.figure(figsize=(12, 10))
        fig = plt.figure()
        m.plotHalls()
        x_hist = [s.car_global_x for s in state_history[m_name]]
        y_hist = [s.car_global_y for s in state_history[m_name]]
        plt.scatter(x_hist, y_hist, s=1, c='r')
        plt.show()
        plt.legend(markerscale=10)
        plt.savefig(f'trajectories_{m.name}.png')
