# typing annotations written for Python 3.8
import numpy as np

from typing import List, Dict, Tuple, Iterable, Any
from hybrid_gym.mode import Mode
from hybrid_gym.transition import Transition


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
        self.modes = {m.name: m for m in mode_object_list}
        self.transitions = {m.name: [] for m in mode_object_list}
        for t in transition_list:
            if t.source in self.transitions:
                self.transitions[t.source].append(t)
        self.mode_list = [m.name for m in mode_object_list]
        self.start_probabilities = list(start_probabilities)

    # reset : () -> state
    def reset(self) -> Tuple[str, Any]:
        new_mode = np.random.choice(self.mode_list, p=self.start_probabilities)
        return new_mode, self.modes[new_mode].reset()

    # step : mode * state * action -> mode * state
    def step(self,
             mode: str,
             state: Any,
             action: np.ndarray
             ) -> Tuple[str, Any]:
        next_state = self.modes[mode].step(state, action)
        for t in self.transitions[mode]:
            if t.guard(next_state):
                return t.jump(next_state)
        return mode, next_state

    # observe : mode * state -> observation
    def observe(self, mode: str, state: Any) -> np.ndarray:
        return self.modes[mode].observe(state)

    # reward : mode * state -> float
    def reward(self, mode: str, state0: Any, action: np.ndarray, state1: Any) -> float:
        return self.modes[mode].reward(state0, action, state1)

    # is_safe : mode * state -> bool
    def is_safe(self, mode: str, state: Any) -> bool:
        return self.modes[mode].is_safe(state)

    # render : mode * state -> ()
    def render(self, mode: str, state: Any) -> None:
        self.modes[mode].render(state)
