'''
Falsify controller for a given mode and pre-region.
'''

import numpy as np
from hybrid_gym.model import Mode, Controller, Transition
from hybrid_gym.synthesis.abstractions import AbstractState, VectorizeWrapper, StateWrapper
from hybrid_gym.falsification.optim import cem
from hybrid_gym.util.test import get_rollout
from typing import List, Callable, Any


def falsify(mode: Mode, transitions: List[Transition], controller: Controller,
            pre: AbstractState, eval_func: Callable[[List[Any]], float],
            max_timesteps: int, num_iter: int, samples_per_iter: int,
            top_samples: int, alpha: float = 0.9, num_init_samples: int = 100,
            print_debug: bool = False) -> List[Any]:

    def f(s):
        state = mode.state_from_vector(s)
        sass, _ = get_rollout(mode, transitions, controller, state, max_timesteps=max_timesteps)
        return eval_func(sass)

    if isinstance(pre, StateWrapper):
        X = pre.abstract_state
    else:
        X = VectorizeWrapper(mode, pre)
    init_samples = [X.sample() for _ in range(num_init_samples)]
    mu = np.mean(init_samples, axis=0)
    sigma = np.std(init_samples, axis=0)

    bad_states = cem(f, X, mu, sigma, num_iter, samples_per_iter, top_samples, alpha, print_debug)
    return [mode.state_from_vector(s) for s in bad_states]


def reward_eval_func(mode: Mode, discount: float = 1.):
    def eval_func(sass):
        reward = 0.
        for sas in reversed(sass):
            reward = mode.reward(*sas) + discount * reward
        return reward
    return eval_func
