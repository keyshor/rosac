import gym
import numpy as np
import joblib
from stable_baselines import A2C, ACER, ACKTR, DDPG, DQN, GAIL, HER, PPO1, PPO2, SAC, TD3, TRPO
from stable_baselines.common.base_class import BaseRLModel
from spectrl.rl.ddpg import DDPG as SpectrlDdpg
from typing import (Iterable, List, Tuple, Dict, Optional,
                    Callable, Generic, NoReturn, TypeVar, Union, Any)
from hybrid_gym.model import Mode, Transition, Controller, StateType

T = TypeVar('T')
NotMethod = Union[T, NoReturn]


class GymEnvWrapper(gym.Env, Generic[StateType]):
    mode: Mode[StateType]
    state: StateType
    observation_space: gym.Space
    action_space: gym.Space
    reset_fn: NotMethod[Callable[[], StateType]]
    reward_fn: NotMethod[Callable[[StateType, np.ndarray, StateType], float]]
    transitions: List[Transition]
    flatten_obs: bool

    def __init__(self,
                 mode: Mode[StateType],
                 transitions: Iterable[Transition],
                 custom_reset: Optional[Callable[[], StateType]] = None,
                 custom_reward: Optional[Callable[[StateType, np.ndarray, StateType], float]] = None,
                 flatten_obs: bool = False
                 ) -> None:
        self.mode = mode
        self.transitions = list(transitions)
        assert all([t.source == mode.name for t in self.transitions])
        self.flatten_obs = flatten_obs
        self.observation_space = gym.spaces.utils.flatten_space(mode.observation_space) \
            if self.flatten_obs else mode.observation_space
        self.action_space = mode.action_space
        self.reset_fn = custom_reset or mode.reset
        self.reward_fn = custom_reward or mode.reward
        self.state = self.reset_fn()
        super().__init__()

    def observe(self) -> Any:
        wrapped_obs = self.mode.observe(self.state)
        return gym.spaces.utils.flatten(self.mode.observation_space, wrapped_obs) \
            if self.flatten_obs else wrapped_obs

    def reset(self) -> np.ndarray:
        self.state = self.reset_fn()
        return self.observe()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        next_state = self.mode.step(self.state, action)
        reward = self.reward_fn(self.state, action, next_state)
        self.state = next_state
        done = not self.mode.is_safe(self.state) \
            or any([t.guard(self.state) for t in self.transitions])
        return self.observe(), reward, done, {}

    def render(self) -> None:
        self.mode.render(self.state)


class GymGoalEnvWrapper(gym.GoalEnv, Generic[StateType]):
    mode: Mode[StateType]
    state: StateType
    observation_space: gym.Space
    action_space: gym.Space
    reset_fn: NotMethod[Callable[[], StateType]]
    reward_fn: NotMethod[Callable[[StateType, np.ndarray, StateType], float]]
    transitions: List[Transition]

    def __init__(self,
                 mode: Mode[StateType],
                 transitions: Iterable[Transition],
                 custom_reset: Optional[Callable[[], StateType]] = None,
                 custom_reward: Optional[Callable[[StateType, np.ndarray, StateType], float]] = None
                 ) -> None:
        self.mode = mode
        self.transitions = list(transitions)
        assert all([t.source == mode.name for t in self.transitions])
        self.observation_space = self.mode.observation_space
        self.action_space = mode.action_space
        self.reset_fn = custom_reset or mode.reset
        self.reward_fn = custom_reward or mode.reward
        self.state = self.reset_fn()
        super().__init__()
        super().reset()

    def reset(self) -> np.ndarray:
        self.state = self.reset_fn()
        return self.mode.observe(self.state)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        next_state = self.mode.step(self.state, action)
        reward = self.reward_fn(self.state, action, next_state)
        self.state = next_state
        done = not self.mode.is_safe(self.state) \
            or any([t.guard(self.state) for t in self.transitions])
        return self.mode.observe(self.state), reward, done, {}

    def render(self) -> None:
        self.mode.render(self.state)

    def compute_reward(self,
                       achieved_goal: np.ndarray,
                       desired_goal: np.ndarray,
                       info: Any,
                       ) -> float:
        return self.mode.compute_reward(achieved_goal, desired_goal, info)


class BaselineCtrlWrapper(Controller):
    model: BaseRLModel

    def __init__(self, model: BaseRLModel) -> None:
        self.model = model

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(observation)
        return action

    def save(self, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(cls,
             path: str,
             algo_name: str = 'td3',
             **kwargs: Dict,
             ) -> Controller:
        if algo_name == 'a2c':
            model: BaseRLModel = A2C.load(path, **kwargs)
        elif algo_name == 'acer':
            model = ACER.load(path, **kwargs)
        elif algo_name == 'acktr':
            model = ACKTR.load(path, **kwargs)
        elif algo_name == 'ddpg':
            model = DDPG.load(path, **kwargs)
        elif algo_name == 'dqn':
            model = DQN.load(path, **kwargs)
        elif algo_name == 'gail':
            model = GAIL.load(path, **kwargs)
        elif algo_name == 'her':
            model = HER.load(path, **kwargs)
        elif algo_name == 'ppo1':
            model = PPO1.load(path, **kwargs)
        elif algo_name == 'ppo2':
            model = PPO2.load(path, **kwargs)
        elif algo_name == 'sac':
            model = SAC.load(path, **kwargs)
        elif algo_name == 'td3':
            model = TD3.load(path, **kwargs)
        elif algo_name == 'trpo':
            model = TRPO.load(path, **kwargs)
        else:
            raise ValueError
        return cls(model)


class SpectrlCtrlWrapper(Controller):
    model: SpectrlDdpg

    def __init__(self, model: SpectrlDdpg) -> None:
        self.model = model

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        return self.model.actor.get_action(observation)

    def save(self, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(cls,
             path: str,
             algo_name: str = 'td3',
             **kwargs: Dict,
             ) -> Controller:
        return cls(joblib.load(path))


class DoneOnSuccessWrapper(gym.Wrapper):
    """
    Reset on success and offsets the reward.
    Useful for GoalEnv.
    """

    def __init__(self, env, reward_offset=1.0):
        super(DoneOnSuccessWrapper, self).__init__(env)
        self.reward_offset = reward_offset

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        done = done or info.get('is_success', False)
        reward += self.reward_offset
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self.env.compute_reward(achieved_goal, desired_goal, info)
        return reward + self.reward_offset
