'''
Common mode selectors for simulation.
'''
import random


from hybrid_gym.model import ModeSelector, Transition, Mode
from typing import Tuple, Any, Iterable


class UniformSelector(ModeSelector):
    '''
    Selects a mode uniformly at random from the list of possible next modes.
    '''

    def __init__(self, modes: Iterable[Mode]) -> None:
        self.modes = list(modes)

    def next_mode(self, transition: Transition, state: Any) -> Tuple[str, bool]:
        mode_name = random.choice(transition.targets)
        return mode_name, False

    def reset(self) -> str:
        return random.choice(self.modes).name


class MaxJumpWrapper(ModeSelector):
    '''
    Wrapper for terminating after a fixed number of jumps.
    '''

    def __init__(self, wrapped_selector: ModeSelector, max_jumps: int) -> None:
        self.wrapped_selector = wrapped_selector
        self.max_jumps = max_jumps
        self.num_jumps = 0

    def next_mode(self, transition: Transition, state: Any) -> Tuple[str, bool]:
        mode_name, done = self.wrapped_selector.next_mode(transition, state)
        self.num_jumps += 1
        done = done or (self.num_jumps > self.max_jumps)
        return mode_name, done

    def reset(self) -> str:
        self.num_jumps = 0
        return self.wrapped_selector.reset()
