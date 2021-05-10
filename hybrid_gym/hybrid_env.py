import numpy as np
import gym

from typing import List, Dict, Tuple, Iterable
from hybrid_gym.model import Mode, Transition, ModeSelector


class HybridAutomaton:
    '''
    Constructs hybrid automaton from modes and transitions.
    Requires all observation spaces to be same gym Boxes.
    Requires all action spaces to be same (can either be Box or Discrete).
    '''
    modes: Dict[str, Mode]
    transitions: Dict[str, List[Transition]]

    def __init__(self, modes: Iterable[Mode], transitions: Iterable[Transition]) -> None:
        self.modes = {m.name: m for m in modes}
        self.transitions = {m.name: [] for m in modes}
        for t in transitions:
            if t.source in self.transitions:
                self.transitions[t.source].append(t)

        mode_list = list(modes)
        self.observation_space = mode_list[0].observation_space
        self.action_space = mode_list[0].action_space
        for mode in mode_list:
            self._check_box_spaces(self.observation_space, mode.observation_space)
            if isinstance(self.action_space, gym.spaces.Box):
                self._check_box_spaces(self.action_space, mode.action_space)
            elif isinstance(self.action_space, gym.spaces.Discrete):
                self._check_discrete_spaces(self.action_space, mode.action_space)

    def _check_box_spaces(self, space1: gym.Space, space2: gym.Space) -> None:
        assert space1.shape == space2.shape

    def _check_discrete_spaces(self, space1: gym.Space, space2: gym.Space) -> None:
        assert space1.n == space2.n


class HybridEnv(gym.Env):

    def __init__(self, automaton: HybridAutomaton, selector: ModeSelector) -> None:
        self.automaton = automaton
        self.selector = selector

        self.observation_space = self.automaton.observation_space
        self.action_space = self.automaton.action_space

    def reset(self) -> np.ndarray:
        mode_name = self.selector.reset()
        self.mode = self.automaton.modes[mode_name]
        self.state = self.mode.end_to_end_reset()
        return self.mode.observe(self.state)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        next_state = self.mode.step(self.state, action)
        reward = 0
        done = False

        self.state = next_state
        for t in self.automaton.transitions[self.mode.name]:
            if t.guard(self.state):
                new_mode, done = self.selector.next_mode(t, self.state)
                if not done:
                    self.state = t.jump(new_mode, self.state)
                    self.mode = self.automaton.modes[new_mode]
                else:
                    reward = 1
                break

        if not self.mode.is_safe(self.state):
            reward = -1
            done = True

        return self.mode.observe(self.state), reward, done, {}

    def render(self) -> None:
        self.mode.render(self.state)
