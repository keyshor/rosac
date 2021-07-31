import os
import sys
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.util.wrappers import BaselineCtrlWrapper, GymEnvWrapper
from hybrid_gym.train.mode_pred import ScipyModePredictor
from hybrid_gym.envs import make_f110_model
from hybrid_gym.util.test import end_to_end_test
from hybrid_gym.hybrid_env import HybridEnv
from hybrid_gym.selectors import UniformSelector, MaxJumpWrapper
from hybrid_gym.falsification.rl_based import dqn_adversary, mcts_adversary

import matplotlib.pyplot as plt

if __name__ == '__main__':
    automaton = make_f110_model(straight_lengths=[10])

    save_path = '.'
    if len(sys.argv) > 1:
        save_path = sys.argv[1]

    controllers = {
        name: BaselineCtrlWrapper.load(
            os.path.join(save_path, name, 'best_model.zip'),
            algo_name='td3',
        )
        for name in automaton.modes
    }

    # mode_predictor = ScipyModePredictor.load(
    #     'mode_predictor.mlp', automaton.observation_space, 'mlp')

    time_limits = {m: 100 for m in automaton.modes}

    # selector = dqn_adversary(automaton, controllers, time_limits, max_jumps=10,
    #                           learning_timesteps=500, policy_kwargs={'layers': [16, 16]})

    selector = mcts_adversary(automaton, controllers, time_limits, max_jumps=10,
                              num_rollouts=150, print_debug=True)

    print('\nEvaluating with MCTS adversary')
    print('Probability of successful completion: {}'.format(
        end_to_end_test(automaton, selector, controllers, time_limits,
                        num_rollouts=10, max_jumps=10, print_debug=True)))
