from hybrid_gym.util.test import end_to_end_test
from hybrid_gym.hybrid_env import HybridAutomaton
from hybrid_gym.model import Controller
from hybrid_gym.selectors import MaxJumpWrapper, UniformSelector
from hybrid_gym.falsification.rl_based import mcts_adversary
from typing import Dict


def random_selector_eval(automaton: HybridAutomaton,
                         controllers: Dict[str, Controller],
                         time_limits: Dict[str, int],
                         max_jumps: int = 10,
                         eval_rollouts: int = 100,
                         print_debug: bool = False,
                         return_steps: bool = False):
    selector = MaxJumpWrapper(UniformSelector(
        [mode for m, mode in automaton.modes.items()]), max_jumps)
    return end_to_end_test(automaton, selector, controllers, time_limits,
                           num_rollouts=eval_rollouts, max_jumps=max_jumps,
                           print_debug=print_debug, return_steps=return_steps)


def mcts_eval(automaton: HybridAutomaton,
              controllers: Dict[str, Controller],
              time_limits: Dict[str, int],
              max_jumps: int = 10,
              mcts_rollouts: int = 500,
              eval_rollouts: int = 100,
              print_debug: bool = False,
              render: bool = False,
              return_steps: bool = False):
    selector = mcts_adversary(automaton, controllers, time_limits, max_jumps=max_jumps,
                              num_rollouts=mcts_rollouts, print_debug=print_debug)
    return end_to_end_test(automaton, selector, controllers, time_limits,
                           num_rollouts=eval_rollouts, max_jumps=max_jumps,
                           print_debug=print_debug, render=render, return_steps=return_steps)
