'''
Synthesis of pre/post conditions using implication counterexamples.
'''

from hybrid_gym.model import Controller
from hybrid_gym.hybrid_env import HybridAutomaton
from hybrid_gym.synthesis.abstractions import AbstractState
from typing import Dict, List, Any, Tuple, Set


class IE:

    def __init__(self, m1: str, m2: str, s1: Any, s2: Any) -> None:
        self.m1 = m1
        self.m2 = m2
        self.s1 = s1
        self.s2 = s2


class CE:

    def __init__(self, m: str, s: Any) -> None:
        self.m = m
        self.s = s


def synthesize(automaton: HybridAutomaton, controllers: Dict[str, Controller],
               pre: Dict[str, AbstractState], max_timesteps: Dict[str, int],
               num_iter: int, n_samples: int, abstract_samples: int = 0) -> List[CE]:
    start_states: Dict[str, List] = {}
    implication_examples: Set[IE] = set()
    counterexamples: List[CE] = []

    for _ in range(num_iter):

        for m in automaton.modes:

            # Initial set of start states sampled randomly
            if m not in start_states:
                start_states[m] = []
                mode = automaton.modes[m]
                for _ in range(n_samples):
                    start_states[m].append(mode.reset())

            # Sample from current estimate of pre
            else:
                for _ in range(abstract_samples):
                    s = pre[m].sample()
                    if s is not None:
                        start_states[m].append(s)

        # Simulate and generate examples
        ies, ces = generate_examples(automaton, controllers, max_timesteps, start_states)
        counterexamples.extend(ces)
        for m in automaton.modes:
            start_states[m] = []
        for ie in ies:
            if not pre[ie.m2].contains(ie.s2):
                implication_examples.add(ie)
            else:
                start_states[ie.m2].append(ie.s2)

        # Extend the pre sets until fixed point
        reached_fp = False
        while not reached_fp:
            reached_fp = True
            remove_ies = []
            for ie in implication_examples:
                if pre[ie.m1].contains(ie.s1):
                    if not pre[ie.m2].contains(ie.s2):
                        pre[ie.m2].extend(ie.s2)
                        start_states[ie.m2].append(ie.s2)
                        reached_fp = False
                    remove_ies.append(ie)
            for ie in remove_ies:
                implication_examples.discard(ie)

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
            state = s1
            step = 0
            done = False

            # Episode loop
            while not done and step <= max_timesteps[m1]:
                obs = mode.observe(state)
                action = controller.get_action(obs)
                state = mode.step(state, action)

                # Check safety
                if not mode.is_safe(state):
                    counterexamples.append(CE(m1, s1))
                    done = True

                # Check guards of transitions out of m1
                for t in automaton.transitions[m1]:
                    if t.guard(state):
                        for m2 in t.targets:
                            s2 = t.jump(m2, state)
                            implication_examples.append(IE(m1, m2, s1, s2))
                        done = True
                        break

                # Increment step count
                step += 1

    return implication_examples, counterexamples
