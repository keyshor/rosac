'''
CounterExample Guided Reinforcement Learning
'''

from hybrid_gym import HybridAutomaton, Mode, Controller
from hybrid_gym.train.single_mode import make_sb_model, train_stable
from hybrid_gym.synthesis.abstractions import AbstractState
from hybrid_gym.synthesis.ice import synthesize
from hybrid_gym.util.wrappers import BaselineCtrlWrapper
from hybrid_gym.falsification.single_mode import falsify
from typing import List, Dict, Any, Iterable, Callable, Optional

import numpy as np
import random
import os


class ResetFunc:
    '''
    Reset function used to sample start states in training
    '''

    def __init__(self, mode: Mode, states: List = [], prob: float = 0.75) -> None:
        self.mode = mode
        self.states = states
        self.prob = prob

    def __call__(self):
        if np.random.binomial(1, self.prob) and len(self.states) > 0:
            return random.choice(self.states)
        else:
            return self.mode.end_to_end_reset()

    def add_states(self, states: Iterable[Any]) -> None:
        self.states.extend(states)


def cegrl(automaton: HybridAutomaton,
          pre: Dict[str, AbstractState],
          time_limits: Dict[str, int],
          algo_name: str = 'td3',
          steps_per_iter: int = 10000,
          num_iter: int = 20,
          num_synth_iter: int = 10,
          n_samples: int = 20,
          abstract_samples: int = 0,
          print_debug: bool = False,
          use_best_model: bool = False,
          save_path: str = '.',
          use_falsification: bool = False,
          num_falsification_iter: int = 100,
          num_falsification_samples: int = 20,
          num_falsification_top_samples: int = 10,
          falsify_func: Optional[Dict[str, Callable[[List[Any]], float]]] = None,
          train_kwargs: Dict[str, Any] = {},
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
            reload_env = train_stable(models[name], mode, automaton.transitions[name],
                                      total_timesteps=steps_per_iter, init_states=reset_funcs[name],
                                      algo_name=algo_name, max_episode_steps=time_limits[name],
                                      save_path=save_path, **train_kwargs)
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
        pre_copy = pre.copy()
        ces = synthesize(automaton, controllers, pre_copy, time_limits, num_synth_iter,
                         n_samples, abstract_samples, print_debug)

        if falsify_func is None:
            # add counterexamples to reset function
            for ce in ces:
                reset_funcs[ce.m].add_states([ce.s])
        else:
            # use falsification to identify bad states
            for (m, pre_m) in pre_copy.items():
                bad_states = falsify(automaton.modes[m], automaton.transitions[m],
                                     controllers[m], pre_copy[m], falsify_func[m], time_limits[m],
                                     num_falsification_iter, num_falsification_samples,
                                     num_falsification_top_samples)
                reset_funcs[m].add_states(bad_states)

    return controllers
