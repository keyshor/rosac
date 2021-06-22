'''
CounterExample Guided Reinforcement Learning
'''

from hybrid_gym import HybridAutomaton, Mode, Controller
from hybrid_gym.train.single_mode import make_sb_model, train_stable
from hybrid_gym.synthesis.abstractions import AbstractState
from hybrid_gym.synthesis.ice import synthesize
from hybrid_gym.util.wrappers import BaselineCtrlWrapper
from typing import List, Dict, Any
from copy import deepcopy

import numpy as np
import random


class ResetFunc:
    '''
    Reset function used to sample start states in training
    '''

    def __init__(self, mode: Mode, states: List = [], prob: float = 0.5) -> None:
        self.mode = mode
        self.states = states
        self.prob = prob

    def __call__(self):
        if np.random.binomial(1, self.prob) and len(self.states) > 0:
            return random.choice(self.states)
        else:
            return self.mode.reset()

    def add_state(self, state: Any) -> None:
        self.states.append(state)


def cegrl(automaton: HybridAutomaton,
          pre: Dict[str, AbstractState],
          max_timesteps: Dict[str, int],
          algo_name: str = 'td3',
          wrapped_algo: str = 'ddpg',  # only relevent to HER
          policy: str = 'MlpPolicy',
          steps_per_iter: int = 10000,
          num_iter: int = 20,
          action_noise_scale: float = 0.1,
          verbose: int = 0,
          num_synth_iter: int = 10,
          n_samples: int = 20,
          abstract_samples: int = 0,
          print_debug: bool = False
          ) -> Dict[str, Controller]:
    '''
    Train policies for all modes
    '''

    # initialize reset functions and RL agents
    reset_funcs = {name: ResetFunc(mode) for (name, mode) in automaton.modes.items()}
    models = {name: make_sb_model(
        mode,
        automaton.transitions[name],
        algo_name=algo_name,
        wrapped_algo=wrapped_algo,
        policy=policy,
        action_noise_scale=action_noise_scale,
        verbose=verbose)
        for (name, mode) in automaton.modes.items()}
    controllers: Dict[str, Controller] = {
        name: BaselineCtrlWrapper(model) for (name, model) in models.items()}

    for i in range(num_iter):
        print('\n**** Iteration {} ****'.format(i))

        # train agents
        for (name, mode) in automaton.modes.items():
            print('\n---- Training controller for mode {} ----'.format(name))
            train_stable(models[name], mode, automaton.transitions[name],
                         total_timesteps=steps_per_iter, init_states=reset_funcs[name],
                         algo_name=algo_name)

        # synthesis
        print('\n---- Running synthesis ----')
        ces = synthesize(automaton, controllers, deepcopy(pre), max_timesteps, num_synth_iter,
                         n_samples, abstract_samples, print_debug)

        # add counterexamples to reset function
        for ce in ces:
            reset_funcs[ce.m].add_state(ce.s)

    return controllers
