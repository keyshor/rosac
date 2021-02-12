# typing annotations written for Python 3.8
import numpy as np
import gym

from typing import TypeVar
from abc import ABCMeta, abstractmethod

StateType = TypeVar('T')


class Mode(metaclass=ABCMeta):
    '''
    Defines an abstract mode.
    '''
    name: str
    action_space: gym.Space
    observation_space: gym.Space

    def __init__(self,
                 name: str,
                 action_space: gym.Space,
                 observation_space: gym.Space,
                 ) -> None:
        self.name = name
        self.action_space = action_space
        self.observation_space = observation_space

    def step(self, state: StateType, action: np.ndarray) -> StateType:
        assert self.action_space.contains(action)
        new_state = self._step_fn(state, action)
        return new_state

    def observe(self, state: StateType) -> np.ndarray:
        obs = self._observation_fn(state)
        assert self.observation_space.contains(obs)
        return obs

    def reward(self, state: StateType, action: np.ndarray, next_state: StateType) -> float:
        assert self.action_space.contains(action)
        return self._reward_fn(state, action, next_state)

    @abstractmethod
    def reset(self) -> StateType:
        '''
        Returns an initial state.
        '''
        pass

    @abstractmethod
    def is_safe(self, state: StateType) -> bool:
        '''
        Checks safety of the given state.
        '''
        pass

    @abstractmethod
    def render(self, state: StateType) -> None:
        '''
        Renders the given state.
        '''
        pass

    @abstractmethod
    def _step_fn(self, state: StateType, action: np.ndarray) -> StateType:
        '''
        Returns new state after taking action in the given state.
        '''
        pass

    @abstractmethod
    def _observation_fn(self, state: StateType) -> np.ndarray:
        '''
        Returns the observation at the given state.
        '''
        pass

    @abstractmethod
    def _reward_fn(self, state: StateType, action: np.ndarray, next_state: StateType) -> float:
        '''
        Returns a reward for the given step.
        '''
        pass

    def vectorize_state(self, state: StateType) -> np.ndarray:
        '''
        Returns a vector representation of the given state.
        '''
        raise NotImplementedError
