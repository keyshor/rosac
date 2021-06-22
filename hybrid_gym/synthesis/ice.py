'''
Synthesis of pre/post conditions using implication counterexamples.
'''

import random
from hybrid_gym.model import Controller
from hybrid_gym.hybrid_env import HybridAutomaton
from hybrid_gym.synthesis.abstractions import AbstractState
from hybrid_gym.util.test import get_rollout
from typing import Dict, List, Any, Tuple


class IE:

    def __init__(self, m1: str, m2: str, s1: Any, s2: Any) -> None:
        self.m1 = m1
        self.m2 = m2
        self.s1 = s1
        self.s2 = s2


class CE:

    def __init__(self, m: str, s: Any, reason: str) -> None:
        self.m = m
        self.s = s
        self.reason = reason


def synthesize(automaton: HybridAutomaton, controllers: Dict[str, Controller],
               pre: Dict[str, AbstractState], max_timesteps: Dict[str, int],
               num_iter: int, n_samples: int, abstract_samples: int = 0,
               print_debug: bool = False) -> List[CE]:
    all_states: Dict[str, List] = {}
    start_states: Dict[str, List] = {}
    implication_examples: List[IE] = []
    counterexamples: List[CE] = []

    for it in range(num_iter):

        for m in automaton.modes:

            # Initial set of start states sampled randomly
            if m not in all_states:
                mode = automaton.modes[m]
                all_states[m] = [mode.reset() for _ in range(n_samples)]

            # Sample from current estimate of pre
            start_states[m] = random.sample(all_states[m], min(n_samples, len(all_states[m])))
            all_states[m] = [s for s in all_states[m] if s not in start_states[m]]
            for _ in range(abstract_samples):
                s = pre[m].sample()
                if s is not None:
                    start_states[m].append(s)

        # Simulate and generate examples
        ies, ces = generate_examples(automaton, controllers, max_timesteps, start_states)
        counterexamples.extend(ces)
        for ie in ies:
            if not pre[ie.m2].contains(ie.s2):
                implication_examples.append(ie)
            else:
                all_states[ie.m2].append(ie.s2)

        # Extend the pre sets until fixed point
        reached_fp = False
        while not reached_fp:
            reached_fp = True
            remove_ies = []
            for ie in implication_examples:
                if pre[ie.m1].contains(ie.s1):
                    if not pre[ie.m2].contains(ie.s2):
                        pre[ie.m2].extend(ie.s2)
                        all_states[ie.m2].append(ie.s2)
                        reached_fp = False
                    remove_ies.append(ie)
            implication_examples = [ie for ie in implication_examples if ie not in remove_ies]

        if print_debug:
            print('\n**** Iteration {} ****'.format(it))
            print('Implication examples generated: {}'.format(len(ies)))
            print('Counterexamples generated: {}'.format(len(ces)))
            print('Implication examples in buffer: {}'.format(len(implication_examples)))
            for m, pre_m in pre.items():
                print('\nPre({}):\n{}'.format(m, str(pre_m)))

    ret_ces = []
    for ce in counterexamples:
        if pre[ce.m].contains(ce.s):
            ret_ces.append(ce)

    return ret_ces


def generate_examples(automaton: HybridAutomaton, controllers: Dict[str, Controller],
                      max_timesteps: Dict[str, int], start_states: Dict[str, List]
                      ) -> Tuple[List[IE], List[CE]]:
    implication_examples = []
    counterexamples = []

    for m1 in automaton.modes:

        mode = automaton.modes[m1]
        controller = controllers[m1]

        # Simulate m1 from different start states
        for s1 in start_states[m1]:

            # Get a rollout using controller for m1
            sarss, info = get_rollout(mode, automaton.transitions[m1],
                                      controller, s1, max_timesteps[m1])

            # generate counterexamples
            if not info['safe']:
                counterexamples.append(CE(m1, s1, 'safety'))
            if info['jump'] is None:
                counterexamples.append(CE(m1, s1, 'liveness'))
            # generate implication examples
            else:
                for m2 in info['jump'].targets:
                    s2 = info['jump'].jump(m2, sarss[-1][-1])
                    implication_examples.append(IE(m1, m2, s1, s2))

    return implication_examples, counterexamples
