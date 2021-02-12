# typing annotations written for Python 3.8
import numpy as np
from typing import List, Tuple, Iterable, Any
from abc import ABCMeta, abstractmethod


class Transition(metaclass=ABCMeta):
    source: str
    targets: List[str]
    probabilities: List[float]

    def __init__(self,
                 source: str,
                 targets: Iterable[str],
                 probabilities: Iterable[float],
                 ) -> None:
        self.source = source
        self.targets = list(targets)
        probs_unnormalized = list(probabilities)
        assert len(self.targets) == len(probs_unnormalized)
        assert all([p >= 0 for p in probs_unnormalized])
        sum_probs = sum(probs_unnormalized)
        if abs(sum_probs - 1.0) < 1e-8:
            self.probabilities = probs_unnormalized
        else:
            self.probabilities = [p / sum_probs for p in probs_unnormalized]

    def jump(self, state: Any) -> Tuple[str, Any]:
        target_mode = np.random.choice(self.targets, p=self.probabilities)
        target_state = self.jump_target(target_mode, state)
        return target_mode, target_state

    @abstractmethod
    def guard(self, state: Any) -> bool:
        '''
        Specifies when this transition can be taken.
        '''
        pass

    @abstractmethod
    def jump_target(self, target: str, state: Any) -> Any:
        '''
        Transforms the state for the target mode.
        '''
        pass
