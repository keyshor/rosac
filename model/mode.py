#!/usr/bin/python

# typing annotations written for Python 3.8
import numpy as np
from typing import Tuple, Iterable, Iterator, Callable, Optional
import itertools

c: Iterator[int] = itertools.count()


# the Mode class just wraps user-provided functions with run-time type checks
class Mode:
    # callable fields marked as optional
    # workaround for https://github.com/python/mypy/issues/9489
    id: str
    state_shape: Tuple[int, ...]
    action_shape: Tuple[int, ...]
    observation_shape: Tuple[int, ...]
    # reset: () -> state
    reset_callable: Optional[Callable[[np.random.Generator], np.ndarray]]
    # step: state * action -> state
    step_callable: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]
    # observe: state -> observation
    observe_callable: Optional[Callable[[np.ndarray], np.ndarray]]
    # reward: state] -> float
    reward_callable: Optional[Callable[[np.ndarray], float]]
    # is_safe: state -> bool
    is_safe_callable: Optional[Callable[[np.ndarray], bool]]
    # is_done: state -> bool
    is_done_callable: Optional[Callable[[np.ndarray], bool]]
    # render: state -> None
    render_callable: Optional[Callable[[np.ndarray], None]]

    def __init__(self,
                 id: str,
                 state_shape: Iterable[int],
                 action_shape: Iterable[int],
                 observation_shape: Iterable[int],
                 reset: Callable[[np.random.Generator], np.ndarray],
                 step: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 observe: Callable[[np.ndarray], np.ndarray],
                 reward: Callable[[np.ndarray], float],
                 is_safe: Callable[[np.ndarray], bool],
                 is_done: Callable[[np.ndarray], bool],
                 render: Callable[[np.ndarray], None]
                 ) -> None:
        self.id = f'{id}_m{next(c)}'
        state_shape = tuple(state_shape)
        action_shape = tuple(action_shape)
        observation_shape = tuple(observation_shape)
        self.reset_callable = reset
        self.step_callable = step
        self.observe_callable = observe
        self.reward_callable = reward
        self.is_safe_callable = is_safe
        self.is_done_callable = is_done
        self.render_callable = render

    # reset: () -> state
    def reset(self, rng: np.random.Generator) -> np.ndarray:
        assert self.reset_callable is not None
        new_state = self.reset_callable(rng)
        assert new_state.dtype.kind == 'f' \
            and new_state.shape == self.state_shape
        return new_state

    # step: state * action -> state
    def step(self, old_state: np.ndarray, action: np.ndarray) -> np.ndarray:
        assert self.step_callable is not None
        assert old_state.dtype.kind == 'f' \
            and old_state.shape == self.state_shape
        assert action.dtype.kind == 'f' \
            and action.shape == self.action_shape
        new_state = self.step_callable(old_state, action)
        assert new_state.dtype.kind == 'f' \
            and new_state.shape == self.state_shape
        return new_state

    # observe: state -> observation
    def observe(self, state: np.ndarray) -> np.ndarray:
        assert self.observe_callable is not None
        assert state.dtype.kind == 'f' \
            and state.shape == self.state_shape
        observation = self.observe_callable(state)
        assert observation.dtype.kind == 'f' \
            and observation.shape == self.observation_shape
        return observation

    # reward: state -> float
    def reward(self, state: np.ndarray) -> float:
        assert self.reward_callable is not None
        assert state.dtype.kind == 'f' \
            and state.shape == self.state_shape
        return self.reward_callable(state)

    # is_safe: state -> bool
    def is_safe(self, state: np.ndarray) -> bool:
        assert self.is_safe_callable is not None
        assert state.dtype.kind == 'f' \
            and state.shape == self.state_shape
        return self.is_safe_callable(state)

    # is_done: state -> bool
    def is_done(self, state: np.ndarray) -> bool:
        assert self.is_done_callable is not None
        assert state.dtype.kind == 'f' \
            and state.shape == self.state_shape
        return self.is_done_callable(state)

    # render: state -> None
    def render(self, state: np.ndarray) -> None:
        assert self.render_callable is not None
        assert state.dtype.kind == 'f' \
            and state.shape == self.state_shape
        self.render_callable(state)
