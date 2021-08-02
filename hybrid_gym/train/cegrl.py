'''
CounterExample Guided Reinforcement Learning
'''

from hybrid_gym import HybridAutomaton, Mode, Controller
from hybrid_gym.train.single_mode import make_sb_model, train_stable
from hybrid_gym.synthesis.abstractions import AbstractState
from hybrid_gym.synthesis.ice import synthesize
from hybrid_gym.util.wrappers import BaselineCtrlWrapper
from typing import List, Dict, Any

import numpy as np
import random
import os
import gym


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
        #return self.mode.reset()

    def add_state(self, state: Any) -> None:
        self.states.append(state)


def cegrl(automaton: HybridAutomaton,
          pre: Dict[str, AbstractState],
          time_limits: Dict[str, int],
          reload_env: gym.Env,
          algo_name: str = 'td3',
          steps_per_iter: int = 10000,
          num_iter: int = 20,
          num_synth_iter: int = 10,
          n_samples: int = 20,
          abstract_samples: int = 0,
          print_debug: bool = False,
          use_best_model: bool = False,
          save_path: str = '.',
          **kwargs
          ) -> Dict[str, Controller]:
    '''
    Train policies for all modes
    '''

    reset_funcs = {name: ResetFunc(mode) for (name, mode) in automaton.modes.items()}
    models = {name: make_sb_model(
        mode,
        automaton.transitions[name],
        algo_name=algo_name,
        max_episode_steps=time_limits[name],
        **kwargs)
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
                         algo_name=algo_name, max_episode_steps=time_limits[name],
                         save_path=save_path)
            if use_best_model:
                ctrl = BaselineCtrlWrapper.load(os.path.join(save_path, name, 'best_model.zip'),
                                                algo_name=algo_name, env=reload_env)
            else:
                controllers[name].save(os.path.join(save_path, name + '.her'))
                ctrl = BaselineCtrlWrapper.load(os.path.join(save_path, name + '.her'),
                                                algo_name=algo_name, env=reload_env)
            if isinstance(ctrl, BaselineCtrlWrapper):
                models[name] = ctrl.model
                controllers[name] = ctrl

        # synthesis
        print('\n---- Running synthesis ----')
        ces = synthesize(automaton, controllers, pre.copy(), time_limits, num_synth_iter,
                         n_samples, abstract_samples, print_debug)

        # add counterexamples to reset function
        for ce in ces:
            reset_funcs[ce.m].add_state(ce.s)

    return controllers
