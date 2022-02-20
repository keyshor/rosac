import gym
import numpy as np
import joblib
from stable_baselines import A2C, ACER, ACKTR, DDPG, DQN, GAIL, HER, PPO1, PPO2, SAC, TD3, TRPO
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines3 import (
    A2C as SB3_A2C, DDPG as SB3_DDPG, DQN as SB3_DQN,
    PPO as SB3_PPO, SAC as SB3_SAC, TD3 as SB3_TD3,
)
from stable_baselines3.common.base_class import BaseAlgorithm
import torch
from spectrl.rl.ddpg import DDPG as SpectrlDdpg
from hybrid_gym.rl.ddpg import DDPG as RlDdpg
from typing import (Iterable, List, Tuple, Dict, Optional,
                    Callable, Generic, NoReturn, Type, TypeVar, Union, Any)
from hybrid_gym.model import Mode, Transition, Controller, StateType
from hybrid_gym.hybrid_env import HybridAutomaton

T = TypeVar('T')
NotMethod = Union[T, NoReturn]


class GymEnvWrapper(gym.Env, Generic[StateType]):
    mode: Mode[StateType]
    automaton: HybridAutomaton
    state: StateType
    observation_space: gym.Space
    action_space: gym.Space
    reset_fn: NotMethod[Callable[[], StateType]]
    reward_fn: NotMethod[Callable[[StateType, np.ndarray, StateType], float]]
    transitions: List[Transition]
    flatten_obs: bool

    def __init__(self,
                 automaton: HybridAutomaton,
                 mode: Mode[StateType],
                 transitions: Iterable[Transition],
                 custom_reset: Optional[Callable[[], StateType]] = None,
                 custom_reward: Optional[Callable[
                     [StateType, np.ndarray, StateType], float]] = None,
                 flatten_obs: bool = False
                 ) -> None:
        self.mode = mode
        self.automaton = automaton
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

    def reset(self) -> Any:
        self.state = self.reset_fn()
        return self.observe()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        next_state = self.mode.step(self.state, action)
        reward = self.reward_fn(self.state, action, next_state)
        self.state = next_state
        is_success, jump_obs = self.compute_info()
        done = not self.mode.is_safe(self.state) or is_success
        return self.observe(), reward, done, {'is_success': is_success, 'jump_obs': jump_obs}

    def render(self, mode: str = 'human') -> None:
        self.mode.render(self.state)

    def compute_info(self):
        is_success = False
        jump_obs = None
        for t in self.transitions:
            if t.guard(self.state):
                is_success = True
                jump_obs = []
                for target in t.targets:
                    jump_state = t.jump(target, self.state)
                    jump_obs.append(
                        (target, self.automaton.modes[target].observe(jump_state)))
                break
        return is_success, jump_obs


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
        is_success = any([t.guard(self.state) for t in self.transitions])
        done = not self.mode.is_safe(self.state) or is_success
        return self.mode.observe(self.state), reward, done, {'is_success': is_success}

    def render(self) -> None:
        self.mode.render(self.state)

    def compute_reward(self,
                       achieved_goal: np.ndarray,
                       desired_goal: np.ndarray,
                       info: Any,
                       ) -> float:
        return self.mode.compute_reward(achieved_goal, desired_goal, info)


class GymMultiEnvWrapper(gym.Env, Generic[StateType]):
    mode_index: int
    state: StateType
    observation_space: gym.Space
    action_space: gym.Space
    mode_info: List[Tuple[
        Mode[StateType],
        List[Transition],
        Callable[[], StateType],
        Callable[[StateType, np.ndarray, StateType], float],
    ]]
    flatten_obs: bool
    rng: np.random.Generator

    def __init__(self,
                 mode_info: Iterable[Tuple[
                     Mode[StateType],
                     Iterable[Transition],
                     Optional[Callable[[], StateType]],
                     Optional[Callable[[StateType, np.ndarray, StateType], float]],
                 ]],
                 flatten_obs: bool = False,
                 rng: np.random.Generator = np.random.default_rng(),
                 ) -> None:
        self.rng = rng
        self.mode_info = [
            (mode, list(trans), custom_reset or mode.reset, custom_reward or mode.reward)
            for (mode, trans, custom_reset, custom_reward) in mode_info
        ]
        assert all([t.source == mode.name
                    for (mode, transitions, _, _) in self.mode_info
                    for t in transitions
                    ])
        self.flatten_obs = flatten_obs
        first_mode, _, _, _ = self.mode_info[0]
        self.observation_space = gym.spaces.utils.flatten_space(first_mode.observation_space) \
            if self.flatten_obs else first_mode.observation_space
        self.action_space = first_mode.action_space
        super().__init__()

    def observe(self) -> Any:
        mode, _, _, _ = self.mode_info[self.mode_index]
        wrapped_obs = mode.observe(self.state)
        return gym.spaces.utils.flatten(mode.observation_space, wrapped_obs) \
            if self.flatten_obs else wrapped_obs

    def reset(self) -> np.ndarray:
        self.mode_index = self.rng.choice(len(self.mode_info))
        mode, _, reset_fn, _ = self.mode_info[self.mode_index]
        self.state = reset_fn()
        return self.observe()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        mode, transitions, _, reward_fn = self.mode_info[self.mode_index]
        next_state = mode.step(self.state, action)
        reward = reward_fn(self.state, action, next_state)
        self.state = next_state
        is_success = any([t.guard(self.state) for t in transitions])
        done = not mode.is_safe(self.state) or is_success
        return self.observe(), reward, done, {'is_success': is_success}

    def render(self) -> None:
        mode, _, _, _ = self.mode_info[self.mode_index]
        mode.render(self.state)


class GymMultiGoalEnvWrapper(gym.GoalEnv, Generic[StateType]):
    mode_index: int
    state: StateType
    observation_space: gym.Space
    action_space: gym.Space
    mode_info: List[Tuple[
        Mode[StateType],
        List[Transition],
        Callable[[], StateType],
        Callable[[StateType, np.ndarray, StateType], float],
    ]]
    rng: np.random.Generator

    def __init__(self,
                 mode_info: Iterable[Tuple[
                     Mode[StateType],
                     Iterable[Transition],
                     Optional[Callable[[], StateType]],
                     Optional[Callable[[StateType, np.ndarray, StateType], float]],
                 ]],
                 rng: np.random.Generator = np.random.default_rng(),
                 ) -> None:
        self.rng = rng
        self.mode_info = [
            (mode, list(trans), custom_reset or mode.reset, custom_reward or mode.reward)
            for (mode, trans, custom_reset, custom_reward) in mode_info
        ]
        assert all([t.source == mode.name
                    for (mode, transitions, _, _) in self.mode_info
                    for t in transitions
                    ])
        first_mode, _, _, _ = self.mode_info[0]
        self.observation_space = first_mode.observation_space
        self.action_space = first_mode.action_space
        super().__init__()
        super().reset()
        self.reset()

    def reset(self) -> Any:
        self.mode_index = self.rng.choice(len(self.mode_info))
        mode, _, reset_fn, _ = self.mode_info[self.mode_index]
        self.state = reset_fn()
        return mode.observe(self.state)

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, Dict]:
        mode, transitions, _, reward_fn = self.mode_info[self.mode_index]
        next_state = mode.step(self.state, action)
        reward = reward_fn(self.state, action, next_state)
        self.state = next_state
        is_success = any([t.guard(self.state) for t in transitions])
        done = not mode.is_safe(self.state) or is_success
        return mode.observe(self.state), reward, done, {'is_success': is_success}

    def render(self) -> None:
        mode, _, _, _ = self.mode_info[self.mode_index]
        mode.render(self.state)

    def compute_reward(self,
                       achieved_goal: np.ndarray,
                       desired_goal: np.ndarray,
                       info: Any,
                       ) -> float:
        mode, _, _, _ = self.mode_info[self.mode_index]
        return mode.compute_reward(achieved_goal, desired_goal, info)


BaselineCtrlWrapperType = TypeVar('BaselineCtrlWrapperType', bound='BaselineCtrlWrapper')


class BaselineCtrlWrapper(Controller):
    model: BaseRLModel

    def __init__(self, model: BaseRLModel) -> None:
        self.model = model

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(observation)
        return action

    def save(self, name: str, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(cls: Type[BaselineCtrlWrapperType],
             path: str,
             env: gym.Env = None,
             algo_name: str = 'td3',
             **kwargs,
             ) -> BaselineCtrlWrapperType:
        if algo_name == 'a2c':
            model: BaseRLModel = A2C.load(path, env=env, **kwargs)
        elif algo_name == 'acer':
            model = ACER.load(path, env=env, **kwargs)
        elif algo_name == 'acktr':
            model = ACKTR.load(path, env=env, **kwargs)
        elif algo_name == 'ddpg':
            model = DDPG.load(path, env=env, **kwargs)
        elif algo_name == 'dqn':
            model = DQN.load(path, env=env, **kwargs)
        elif algo_name == 'gail':
            model = GAIL.load(path, env=env, **kwargs)
        elif algo_name == 'her':
            model = HER.load(path, env=env, **kwargs)
        elif algo_name == 'ppo1':
            model = PPO1.load(path, env=env, **kwargs)
        elif algo_name == 'ppo2':
            model = PPO2.load(path, env=env, **kwargs)
        elif algo_name == 'sac':
            model = SAC.load(path, env=env, **kwargs)
        elif algo_name == 'td3':
            model = TD3.load(path, env=env, **kwargs)
        elif algo_name == 'trpo':
            model = TRPO.load(path, env=env, **kwargs)
        else:
            raise ValueError
        return cls(model)


Sb3CtrlWrapperType = TypeVar('Sb3CtrlWrapperType', bound='Sb3CtrlWrapper')


class Sb3CtrlWrapper(Controller):
    model: BaseAlgorithm

    def __init__(self, model: BaseAlgorithm) -> None:
        self.model = model

    def get_action(self, observation: Any) -> np.ndarray:
        m_obs_sp = self.model.observation_space
        if m_obs_sp:
            # print(np.can_cast(observation.dtype, m_obs_sp.dtype))
            # print(observation.shape == m_obs_sp.shape)
            # print(np.all(observation >= m_obs_sp.low))
            # print(np.all(observation <= m_obs_sp.high))
            # print(observation.shape)
            # print(m_obs_sp.shape)
            assert m_obs_sp.contains(observation)
        action, _ = self.model.predict(observation, deterministic=True)
        return action

    def save(self, name: str, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(cls: Type[Sb3CtrlWrapperType],
             path: str,
             algo_name: str,
             **kwargs,
             ) -> Sb3CtrlWrapperType:
        if algo_name == 'a2c':
            model: BaseAlgorithm = SB3_A2C.load(path, **kwargs)
        elif algo_name == 'ddpg':
            model = SB3_DDPG.load(path, **kwargs)
        elif algo_name == 'dqn':
            model = SB3_DQN.load(path, **kwargs)
        elif algo_name == 'ppo':
            model = SB3_PPO.load(path, **kwargs)
        elif algo_name == 'sac':
            model = SB3_SAC.load(path, **kwargs)
        elif algo_name == 'td3':
            model = SB3_TD3.load(path, **kwargs)
        else:
            raise ValueError
        return cls(model=model)


SpectrlCtrlWrapperType = TypeVar('SpectrlCtrlWrapperType', bound='SpectrlCtrlWrapper')


class SpectrlCtrlWrapper(Controller):
    model: SpectrlDdpg

    def __init__(self, model: SpectrlDdpg) -> None:
        self.model = model

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        return self.model.actor.get_action(observation)

    def save(self, name: str, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(cls: Type[SpectrlCtrlWrapperType],
             path: str,
             algo_name: str = 'td3',
             **kwargs,
             ) -> SpectrlCtrlWrapperType:
        return cls(joblib.load(path))


DdpgCtrlWrapperType = TypeVar('DdpgCtrlWrapperType', bound='DdpgCtrlWrapper')


class DdpgCtrlWrapper(Controller):
    model: RlDdpg

    def __init__(self, model: RlDdpg) -> None:
        self.model = model

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        return self.model.actor.get_action(observation)

    def save(self, name: str, path: str) -> None:
        torch.save(self.model, path)

    @classmethod
    def load(cls: Type[DdpgCtrlWrapperType],
             path: str,
             algo_name: str = 'ddpg',
             **kwargs,
             ) -> DdpgCtrlWrapperType:
        return torch.load(path, **kwargs)


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
