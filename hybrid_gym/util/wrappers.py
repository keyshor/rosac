import gym
import numpy as np
from stable_baselines import A2C, ACER, ACKTR, DDPG, DQN, GAIL, HER, PPO1, PPO2, SAC, TD3, TRPO
from stable_baselines.common.base_class import BaseRLModel
from typing import (Iterable, List, Tuple, Dict, Optional,
                    Callable, Generic, NoReturn, TypeVar, Union)
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

    def __init__(self,
                 mode: Mode[StateType],
                 transitions: Iterable[Transition],
                 custom_reset: Optional[Callable[[], StateType]] = None,
                 custom_reward: Optional[Callable[[StateType, np.ndarray, StateType], float]] = None
                 ) -> None:
        self.mode = mode
        self.transitions = list(transitions)
        assert all([t.source == mode.name for t in self.transitions])
        self.observation_space = mode.observation_space
        self.action_space = mode.action_space
        self.reset_fn = custom_reset or mode.reset
        self.reward_fn = custom_reward or mode.reward
        self.state = self.reset_fn()
        super().__init__()

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
        self.observation_space = gym.spaces.Dict({
            'observation': self.mode.observation_space,
            'achieved_goal': self.mode.goal_space,
            'desired_goal': self.mode.goal_space,
        })
        self.action_space = mode.action_space
        self.reset_fn = custom_reset or mode.reset
        self.reward_fn = custom_reward or mode.reward
        self.state = self.reset_fn()
        super().__init__()

    def observe_with_goal(self):
        return {
            'observation': self.mode.observe(self.state),
            'achieved_goal': self.mode.achieved_goal(self.state),
            'desired_goal': self.mode.desired_goal(self.state),
        }

    def reset(self) -> np.ndarray:
        self.state = self.reset_fn()
        return self.observe_with_goal()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        next_state = self.mode.step(self.state, action)
        reward = self.reward_fn(self.state, action, next_state)
        self.state = next_state
        done = not self.mode.is_safe(self.state) \
            or any([t.guard(self.state) for t in self.transitions])
        return self.observe_with_goal(), reward, done, {}

    def render(self) -> None:
        self.mode.render(self.state)


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
