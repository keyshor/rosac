'''
Common mode selectors for simulation.
'''
import random


from hybrid_gym.model import ModeSelector, Transition, Mode
from typing import Tuple, Any, Iterable, List


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

class FixedSequenceSelector(ModeSelector):
    '''
    Selects a fixed sequence of modes.
    '''
    mode_list: List[str]
    index: int
    loop_at_end: bool

    def __init__(self,
                 mode_list: Iterable[Mode],
                 loop_at_end: bool = False,
                 ) -> None:
        self.mode_list = [m.name for m in mode_list]
        self.index = 0
        self.loop_at_end = loop_at_end

    def next_mode(self, transition: Transition, state: Any) -> Tuple[str, bool]:
        try:
            assert transition.source == self.mode_list[self.index] and \
                self.mode_list[self.index + 1] in transition.targets
            self.index += 1
            return self.mode_list[self.index], False
        except IndexError:
            return (self.reset(), False) if self.loop_at_end else \
                (self.mode_list[self.index], True)

    def reset(self) -> str:
        self.index = 0
        return self.mode_list[0]
