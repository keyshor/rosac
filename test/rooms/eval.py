import os
import sys
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.util.wrappers import Sb3CtrlWrapper, GymEnvWrapper
from hybrid_gym.train.mode_pred import ScipyModePredictor
from hybrid_gym.envs import make_rooms_model
from hybrid_gym.eval import mcts_eval, random_selector_eval
from hybrid_gym.hybrid_env import HybridEnv
from hybrid_gym.util.io import parse_command_line_options
from hybrid_gym.rl.ars import NNPolicy

import matplotlib.pyplot as plt


def _get_suffix(use_best):
    if use_best:
        return ''
    return '_final'


if __name__ == '__main__':

    flags = parse_command_line_options()
    automaton = make_rooms_model()

    controllers = {
        name: NNPolicy.load(name + _get_suffix(flags['best']), flags['path'], flags['gpu'])
        for name in automaton.modes
    }

    time_limits = {m: 25 for m in automaton.modes}

    if flags['falsify']:
        print('\nEvaluating with MCTS adversary')
        prob = mcts_eval(automaton, controllers, time_limits, max_jumps=15,
                         mcts_rollouts=500, eval_rollouts=100, print_debug=True)
    else:
        print('\nEvaluating with random adversary')
        prob = random_selector_eval(automaton, controllers, time_limits, max_jumps=15,
                                    eval_rollouts=100, print_debug=True)

    print('Probability of successful completion: {}'.format(prob))
