import os
import sys
sys.path.append(os.path.join('..', '..'))  # nopep8

from hybrid_gym.train.single_mode import train_stable
from hybrid_gym.envs import make_f110_model


if __name__ == '__main__':
    automaton = make_f110_model(straight_lengths=[10])
    for (name, mode) in automaton.modes.items():
        controller = train_stable(mode, automaton.transitions[name], algo_name='td3', verbose=1)
        controller.save(f'{name}.td3')
