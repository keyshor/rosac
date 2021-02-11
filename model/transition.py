#!/usr/bin/python

# typing annotations written for Python 3.8
import numpy as np
from typing import List, Dict, Tuple, Iterable, Callable, Optional


class Transition:
    # callable fields marked as optional
    # workaround for https://github.com/python/mypy/issues/9489
    state_shape: Tuple[int, ...]
    guard_callable: Optional[Callable[[np.ndarray], bool]]
    source: str
    targets: List[str]
    probabilities: List[float]
    jump_callables: Dict[str, Callable[[np.ndarray], np.ndarray]]

    def __init__(self,
                 state_shape: Iterable[int],
                 guard_callable: Callable[[np.ndarray], bool],
                 source: str,
                 targets: Iterable[str],
                 jump_callables: Iterable[Callable[[np.ndarray], np.ndarray]],
                 probabilities: Iterable[float],
                 ) -> None:
        self.state_shape = tuple(state_shape)
        self.guard_callable = guard_callable
        self.source = source
        self.targets = list(targets)
        jump_callable_list = list(jump_callables)
        probabilities_unnormalized = list(probabilities)
        assert len(self.targets) == len(jump_callable_list) \
            == len(probabilities_unnormalized)
        self.jump_callables = {
            self.targets[i]: jump_callable_list[i]
            for i in range(len(self.targets))
        }
        assert all([p >= 0 for p in probabilities_unnormalized])
        sum_probs = sum(probabilities_unnormalized)
        if abs(sum_probs - 1.0) < 0.001:
            self.probabilities = probabilities_unnormalized
        else:
            self.probabilities = [p / sum_probs
                                  for p in probabilities_unnormalized]

    def guard(self, state: np.ndarray) -> bool:
        assert self.guard_callable is not None
        assert state.dtype.kind == 'f' \
            and state.shape == self.state_shape
        return self.guard_callable(state)

    # jump: state -> str * state
    def jump(self, old_state: np.ndarray, rng: np.random.Generator) \
            -> Tuple[str, np.ndarray]:
        assert old_state.dtype.kind == 'f' \
            and old_state.shape == self.state_shape
        new_mode = rng.choice(self.targets, p=self.probabilities)
        new_state = self.jump_callables[new_mode](old_state)
        assert new_state.dtype.kind == 'f' \
            and new_state.shape == self.state_shape
        return new_mode, new_state
