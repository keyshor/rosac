from abc import ABCMeta, abstractmethod
from typing import Any, Optional
from copy import deepcopy

import numpy as np


class AbstractState(metaclass=ABCMeta):
    '''
    Base class for abstract domains.
    '''

    @abstractmethod
    def contains(self, state: Any) -> bool:
        pass

    @abstractmethod
    def extend(self, state: Any) -> None:
        pass

    def sample(self) -> Any:
        '''
        Sampling is optional
        '''
        return NotImplementedError

    def copy(self):
        return deepcopy(self)


class Box(AbstractState):
    low: Any  # Hack to make mypy accept np.minimum and np.maximim.
    high: Any

    def __init__(self, low: Optional[np.ndarray] = None, high: Optional[np.ndarray] = None) -> None:
        self.low = low
        self.high = high

    def contains(self, state: np.ndarray) -> bool:
        if self.low is not None and self.high is not None:
            if np.all(self.low <= state) and np.all(self.high >= state):
                return True
        return False

    def extend(self, state: np.ndarray) -> None:
        if self.low is not None:
            self.low = np.minimum(self.low, state)
        else:
            self.low = state.copy()
        if self.high is not None:
            self.high = np.maximum(self.high, state)
        else:
            self.high = state.copy()

    def sample(self) -> Optional[np.ndarray]:
        if self.low is not None and self.high is not None:
            return np.random.uniform(self.low, self.high)
        else:
            return None

    def __str__(self):
        low = 'None'
        high = 'None'
        if self.low is not None:
            low = self.low.tolist()
        if self.high is not None:
            high = self.high.tolist()
        return 'Low: {}\nHigh: {}'.format(low, high)


class VectorizeWrapper(AbstractState):
    '''
    Defines an abstract state in the space of vectorized states from
        an abstract state in the original state space.
    '''

    def __init__(self, mode, abstract_state) -> None:
        self.mode = mode
        self.abstract_state = abstract_state

    def contains(self, state: np.ndarray) -> bool:
        return self.abstract_state.contains(self.mode.state_from_vector(state))

    def extend(self, state: np.ndarray) -> None:
        self.abstract_state.extend(self.mode.state_from_vector(state))

    def sample(self) -> Optional[np.ndarray]:
        state = self.abstract_state.sample()
        if state is not None:
            state = self.mode.vectorize_state(state)
        return state

    def copy(self):
        return StateWrapper(self.mode, self.abstract_state.copy())


class StateWrapper(AbstractState):
    '''
    Defines an abstract state in the original state space from
        an abstract state in the space of vectorized states.
    '''

    def __init__(self, mode, abstract_state) -> None:
        self.mode = mode
        self.abstract_state = abstract_state

    def contains(self, state) -> bool:
        return self.abstract_state.contains(self.mode.vectorize_state(state))

    def extend(self, state) -> None:
        self.abstract_state.extend(self.mode.vectorize_state(state))

    def sample(self) -> Any:
        state = self.abstract_state.sample()
        if state is not None:
            state = self.mode.state_from_vector(state)
        return state

    def copy(self):
        return StateWrapper(self.mode, self.abstract_state.copy())

    def __str__(self):
        return self.abstract_state.__str__()
