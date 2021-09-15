import os
import sys
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.util.wrappers import Sb3CtrlWrapper, GymEnvWrapper
from hybrid_gym.train.mode_pred import ScipyModePredictor
from hybrid_gym.envs import make_rooms_model
from hybrid_gym.util.test import end_to_end_test
from hybrid_gym.hybrid_env import HybridEnv
from hybrid_gym.selectors import UniformSelector, MaxJumpWrapper
from hybrid_gym.falsification.rl_based import dqn_adversary, mcts_adversary
from hybrid_gym.util.io import parse_command_line_options

import matplotlib.pyplot as plt

if __name__ == '__main__':

    flags = parse_command_line_options()
    automaton = make_rooms_model()

    controllers = {
        name: Sb3CtrlWrapper.load(
            os.path.join(flags['path'], name, 'best_model'),
            algo_name='td3',
        )
        for name in automaton.modes
    }

    time_limits = {m: 25 for m in automaton.modes}

    if flags['falsify']:
        selector = mcts_adversary(automaton, controllers, time_limits, max_jumps=10,
                                  num_rollouts=500, print_debug=True)
    else:
        selector = MaxJumpWrapper(UniformSelector(
            [mode for m, mode in automaton.modes.items()]), 10)

    # selector = dqn_adversary(automaton, controllers, time_limits, max_jumps=10,
    #                          learning_timesteps=500, policy_kwargs={'layers': [16, 16]})

    print('\nEvaluating with MCTS adversary')
    print('Probability of successful completion: {}'.format(
        end_to_end_test(automaton, selector, controllers, time_limits,
                        num_rollouts=100, max_jumps=10, print_debug=True,
                        render=flags['render'])))
