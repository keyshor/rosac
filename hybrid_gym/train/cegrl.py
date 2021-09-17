'''
CounterExample Guided Reinforcement Learning
'''

from hybrid_gym import HybridAutomaton, Mode, Controller
from hybrid_gym.train.single_mode import make_sb3_model, train_sb3, make_ars_model, learn_ars_model
from hybrid_gym.synthesis.abstractions import AbstractState
from hybrid_gym.synthesis.ice import synthesize
from hybrid_gym.util.wrappers import Sb3CtrlWrapper
from hybrid_gym.falsification.single_mode import falsify
from hybrid_gym.rl.ars import NNPolicy
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
          mode_groups: List[List[Mode]] = [],
          print_debug: bool = False,
          use_best_model: bool = False,
          save_path: str = '.',
          algo_name: str = 'td3',
          num_iter: int = 20,
          num_synth_iter: int = 10,
          n_synth_samples: int = 50,
          abstract_synth_samples: int = 0,
          num_falsification_iter: int = 100,
          num_falsification_samples: int = 20,
          num_falsification_top_samples: int = 10,
          falsify_func: Optional[Dict[str, Callable[[List[Any]], float]]] = None,
          **kwargs
          ) -> Dict[str, Controller]:
    '''
    Train policies for all modes
    '''

    reset_funcs = {name: ResetFunc(mode) for (name, mode) in automaton.modes.items()}

    # Add each mode into its own group if no grouping is given
    if len(mode_groups) == 0:
        mode_groups = [[mode] for _, mode in automaton.modes.items()]
    group_names = [[mode.name for mode in modes] for modes in mode_groups]

    # map modes to their group numbers
    group_map = {}
    for g in range(len(mode_groups)):
        for name in group_names[g]:
            group_map[name] = g

    # all necessary information about all groups
    group_info = [[(mode, automaton.transitions[mode.name], reset_funcs[mode.name], None)
                   for mode in modes] for modes in mode_groups]

    # create one model and one controller for each group
    if algo_name == 'ars':
        models = [make_ars_model(**kwargs)
                  for _ in mode_groups]
        controllers: List[Controller] = [model.nn_policy for model in models]
    else:
        models = [make_sb3_model(
            group_info[g],
            algo_name=algo_name,
            max_episode_steps=time_limits[group_names[g][0]],
            **kwargs)
            for g in range(len(mode_groups))]
        controllers = [Sb3CtrlWrapper(model) for model in models]

    for i in range(num_iter):
        print('\n**** Iteration {} ****'.format(i))

        # train agents
        for g in range(len(mode_groups)):
            print('\n---- Training controller for modes {} ----'.format(group_names[g]))
            if algo_name == 'ars':
                learn_ars_model(models[g], list(group_info[g]),
                                save_path=save_path, verbose=print_debug)
            else:
                train_sb3(models[g], group_info[g],
                          algo_name=algo_name, save_path=save_path,
                          max_episode_steps=time_limits[group_names[g][0]],
                          **kwargs)

            if use_best_model:
                if algo_name == 'ars':
                    nn_policy = NNPolicy.load(group_names[g][0], save_path, **kwargs)
                    models[g].nn_policy = nn_policy
                else:
                    ctrl = Sb3CtrlWrapper.load(
                        os.path.join(save_path, group_names[g][0], 'best_model.zip'),
                        algo_name=algo_name,  # env=reload_env,
                    )
                    models[g].set_parameters(ctrl.model.get_parameters())

        if algo_name == 'ars':
            controllers = [model.nn_policy for model in models]

        # synthesis
        print('\n---- Running synthesis ----')
        pre_copy = {name: astate.copy() for name, astate in pre.items()}
        mode_controllers = {name: controllers[g] for name, g in group_map.items()}
        ces = synthesize(automaton, mode_controllers, pre_copy, time_limits, num_synth_iter,
                         n_synth_samples, abstract_synth_samples, print_debug)

        if falsify_func is not None:
            # use falsification to identify bad states
            for (m, pre_m) in pre_copy.items():
                bad_states = falsify(automaton.modes[m], automaton.transitions[m],
                                     mode_controllers[m], pre_copy[m], falsify_func[m],
                                     time_limits[m], num_falsification_iter,
                                     num_falsification_samples,
                                     num_falsification_top_samples)
                reset_funcs[m].add_states(bad_states)
        else:
            # add counterexamples to reset function
            for ce in ces:
                reset_funcs[ce.m].add_states([ce.s])

    return mode_controllers
