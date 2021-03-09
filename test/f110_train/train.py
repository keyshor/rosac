import os
import sys
sys.path.append(os.path.join('..', '..'))
from hybrid_gym.envs.f110.hybrid_env import make_hybrid_env
from hybrid_gym.train_single_mode.stable_baselines_wrapper import train_stable

if __name__ == '__main__':
    env = make_hybrid_env(straight_lengths=[10])
    for (name, mode) in env.automaton.modes.items():
        controller = train_stable(mode, env.automaton.transitions[name], algo_name='td3', verbose=1)
        controller.save(f'{name}.td3')
