# typing annotations written for Python 3.8
import numpy as np
from typing import List, Dict, Tuple, Iterable
from mode import Mode
from transition import Transition


class HybridEnv:
    modes: Dict[str, Mode]
    transitions: Dict[str, List[Transition]]
    mode_list: List[str]
    start_probabilities: List[float]

    def __init__(self,
                 modes: Iterable[Mode],
                 start_probabilities: Iterable[float],
                 transitions: Iterable[Transition]
                 ) -> None:
        mode_object_list = list(modes)
        transition_list = list(transitions)
        self.modes = {m.id: m for m in mode_object_list}
        self.transitions = {
            m.id: [t for t in transition_list if t.source == m.id]
            for m in mode_object_list
        }
        self.mode_list = [m.id for m in mode_object_list]
        self.start_probabilities = list(start_probabilities)

    # reset : () -> mode * state
    def reset(self, rng: np.random.Generator) -> Tuple[str, np.ndarray]:
        new_mode = rng.choice(self.mode_list, p=self.start_probabilities)
        return new_mode, self.modes[new_mode].reset(rng)

    # step : mode * state * action -> mode * state
    def step(self,
             old_mode: str,
             old_state: np.ndarray,
             action: np.ndarray,
             rng: np.random.Generator
             ) -> Tuple[str, np.ndarray]:
        middle_state = self.modes[old_mode].step(old_state, action)
        for t in self.transitions[old_mode]:
            if t.guard(middle_state):
                return t.jump(middle_state, rng)
        return old_mode, middle_state

    # observe : mode * state -> observation
    def observe(self, mode: str, state: np.ndarray) -> np.ndarray:
        return self.modes[mode].observe(state)

    # reward : mode * state -> float
    def reward(self, mode: str, state: np.ndarray) -> float:
        return self.modes[mode].reward(state)

    # is_safe : mode * state -> bool
    def is_safe(self, mode: str, state: np.ndarray) -> bool:
        return self.modes[mode].is_safe(state)

    # is_done : mode * state -> bool
    def is_done(self, mode: str, state: np.ndarray) -> bool:
        return self.modes[mode].is_done(state)

    # render : mode * state -> ()
    def render(self, mode: str, state: np.ndarray) -> None:
        self.modes[mode].render(state)
