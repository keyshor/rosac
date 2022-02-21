import numpy as np
import gym
from matplotlib.axes import Axes

from typing import TypeVar, Generic, List, Any, Iterable, Tuple, Callable
from abc import ABCMeta, abstractmethod
from hybrid_gym.util.common import obs_shape, space_shape
from hybrid_gym.synthesis.abstractions import AbstractState, Box, StateWrapper

StateType = TypeVar('StateType')


class Mode(Generic[StateType], metaclass=ABCMeta):
    '''
    Defines an abstract mode of the hybrid automaton.
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

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        if isinstance(self.action_space, gym.spaces.Box):
            action_any: Any = action
            return np.clip(a=action_any, a_min=self.action_space.low, a_max=self.action_space.high)
        return action

    def step(self, state: StateType, action: np.ndarray) -> StateType:
        assert action.shape == self.action_space.shape
        new_state = self._step_fn(state, self.clip_action(action))
        return new_state

    def observe(self, state: StateType) -> Any:
        obs = self._observation_fn(state)
        assert self.observation_space.contains(obs), \
            f'obs.shape = {obs_shape(obs)}, space.shape = {space_shape(self.observation_space)}'
        return obs

    def reward(self, state: StateType, action: np.ndarray, next_state: StateType) -> float:
        return self._reward_fn(state, self.clip_action(action), next_state)

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
    def _observation_fn(self, state: StateType) -> Any:
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

    @abstractmethod
    def vectorize_state(self, state: StateType) -> np.ndarray:
        '''
        Returns a vector representation of the given state.
        '''
        pass

    @abstractmethod
    def state_from_vector(self, vec: np.ndarray) -> StateType:
        '''
        Returns state from a given vector.
        '''
        pass

    def end_to_end_reset(self) -> StateType:
        '''
        Allows change in distribution in end-to-end testing.
        '''
        return self.reset()

    def compute_reward(self,
                       achieved_goal: np.ndarray,
                       desired_goal: np.ndarray,
                       info: Any,
                       ) -> float:
        '''
        for compatibility with GoalEnv.
        '''
        raise NotImplementedError

    def plot_state_iterable(self, ax: Axes, sts: Iterable[StateType]) -> None:
        '''
        plot a set of states.
        '''
        raise NotImplementedError

    def normalize_exit_state(self, obs: np.ndarray) -> np.ndarray:
        '''
        Normalizes observations of terminal states in the mode.
        '''
        return obs

    def get_init_pre(self) -> AbstractState:
        '''
        return an AbstractState that gives lower and upper bounds
        on each component of the starting state vector
        '''
        state_shape = self.vectorize_state(self.reset()).shape
        low = np.ones(shape=state_shape) * np.inf
        high = -np.ones(shape=state_shape) * np.inf
        for _ in range(100):
            v = self.vectorize_state(self.reset())
            low = np.minimum(low, v)
            high = np.maximum(high, v)
        return StateWrapper(self, Box(low=low, high=high))


class Transition(metaclass=ABCMeta):
    source: str
    targets: List[str]

    def __init__(self, source: str, targets: Iterable[str]) -> None:
        self.source = source
        self.targets = list(targets)

    @abstractmethod
    def guard(self, state: Any) -> bool:
        '''
        Specifies when this transition can be taken.
        '''
        pass

    @abstractmethod
    def jump(self, target: str, state: Any) -> Any:
        '''
        Transforms the state for the target mode.
        '''
        pass


class ModeSelector(metaclass=ABCMeta):
    '''
    Selects the next mode during a transition.
    Can be stateful, tracking the history of transitions.
    '''

    @abstractmethod
    def next_mode(self, transition: Transition, state: Any) -> Tuple[str, bool]:
        '''
        Returns the mode (name) to switch to and whether the simulation is done.
        '''
        pass

    @abstractmethod
    def reset(self) -> str:
        '''
        Returns an initial mode (name).
        '''
        pass


ControllerType = TypeVar('ControllerType', bound='Controller')


class Controller(metaclass=ABCMeta):
    '''
    Abstract class for controller.
    '''

    @abstractmethod
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        pass

    def reset(self) -> None:
        pass

    @abstractmethod
    def save(self, name: str, path: str) -> None:
        pass

    def get_value_fn(self) -> Callable[[np.ndarray, bool], float]:
        '''
        Returns observation based value function such as the one learned by SAC.
        Second argument allows for optionally setting the function to be deterministic
            in cases where the policy is stochastic.
        '''
        raise NotImplementedError


class ModePredictor(metaclass=ABCMeta):
    '''
    Abstract class for mode predictor.
    '''

    @abstractmethod
    def get_mode(self, observation: np.ndarray) -> str:
        pass

    def reset(self) -> None:
        pass
