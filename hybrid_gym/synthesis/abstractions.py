from abc import ABCMeta, abstractmethod
from typing import Any, Union

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


class Box(AbstractState):
    low: Union[np.ndarray, None]
    high: Union[np.ndarray, None]

    def __init__(self) -> None:
        self.low = None
        self.high = None

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

    def sample(self) -> Union[np.ndarray, None]:
        if self.low is not None and self.high is not None:
            return np.random.uniform(self.low, self.high)
        else:
            return None
