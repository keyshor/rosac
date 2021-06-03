import gym
import numpy as np
from stable_baselines import A2C, ACER, ACKTR, DDPG, DQN, GAIL, HER, PPO1, PPO2, SAC, TD3, TRPO
from stable_baselines.common.base_class import BaseRLModel
from typing import (Iterable, List, Tuple, Dict, Optional,
                    Union, Callable, NoReturn, Generic, TypeVar)
from stable_baselines.ddpg.noise import NormalActionNoise
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
             algo_name: str = 'td3'
             ) -> Controller:
        if algo_name == 'a2c':
            model: BaseRLModel = A2C.load(path)
        elif algo_name == 'acer':
            model = ACER.load(path)
        elif algo_name == 'acktr':
            model = ACKTR.load(path)
        elif algo_name == 'ddpg':
            model = DDPG.load(path)
        elif algo_name == 'dqn':
            model = DQN.load(path)
        elif algo_name == 'gail':
            model = GAIL.load(path)
        elif algo_name == 'her':
            model = HER.load(path)
        elif algo_name == 'ppo1':
            model = PPO1.load(path)
        elif algo_name == 'ppo2':
            model = PPO2.load(path)
        elif algo_name == 'sac':
            model = SAC.load(path)
        elif algo_name == 'td3':
            model = TD3.load(path)
        elif algo_name == 'trpo':
            model = TRPO.load(path)
        else:
            raise ValueError
        return cls(model)


def make_sb_model(mode: Mode[StateType],
                  transitions: Iterable[Transition],
                  algo_name: str = 'td3',
                  wrapped_algo: str = 'ddpg',  # only relevent to HER
                  policy: str = 'MlpPolicy',
                  action_noise_scale: float = 0.1,
                  verbose: int = 0
                  ) -> BaseRLModel:
    env = GymEnvWrapper(mode, transitions)
    goal_env = GymGoalEnvWrapper(mode, transitions)
    action_shape = mode.action_space.shape
    ddpg_action_noise = NormalActionNoise(
        mean=np.zeros(action_shape),
        sigma=action_noise_scale * np.ones(action_shape)
    )
    if algo_name == 'a2c':
        model: BaseRLModel = A2C(policy, env, verbose=verbose)
    elif algo_name == 'acer':
        model = ACER(policy, env, verbose=verbose)
    elif algo_name == 'acktr':
        model = ACKTR(policy, env, verbose=verbose)
    elif algo_name == 'ddpg':
        model = DDPG(policy, env, action_noise=ddpg_action_noise, verbose=verbose)
    elif algo_name == 'dqn':
        model = DQN(policy, env, verbose=verbose)
    elif algo_name == 'gail':
        model = GAIL(policy, env, verbose=verbose)
    elif algo_name == 'her' and wrapped_algo == 'ddpg':
        model = HER(policy, goal_env, DDPG, verbose=verbose)
    elif algo_name == 'her' and wrapped_algo == 'dqn':
        model = HER(policy, goal_env, DQN, verbose=verbose)
    elif algo_name == 'her' and wrapped_algo == 'sac':
        model = HER(policy, goal_env, SAC, verbose=verbose)
    elif algo_name == 'her' and wrapped_algo == 'td3':
        model = HER(policy, goal_env, TD3, verbose=verbose)
    elif algo_name == 'ppo1':
        model = PPO1(policy, env, verbose=verbose)
    elif algo_name == 'ppo2':
        model = PPO2(policy, env, verbose=verbose)
    elif algo_name == 'sac':
        model = SAC(policy, env, verbose=verbose)
    elif algo_name == 'td3':
        model = TD3(policy, env, action_noise=ddpg_action_noise, verbose=verbose)
    elif algo_name == 'trpo':
        model = TRPO(policy, env, verbose=verbose)
    else:
        raise ValueError
    return model


def train_stable(model, mode, transitions, total_timesteps=1000,
                 init_states: Optional[Callable[[], StateType]] = None,
                 reward_fn: Optional[Callable[[StateType, np.ndarray, StateType], float]] = None,
                 algo_name='td3'):
    env = GymEnvWrapper(mode, transitions, init_states, reward_fn)
    if algo_name == 'her':
        env = GymGoalEnvWrapper(mode, transitions, init_states, reward_fn)
    model.set_env(env)
    model.learn(total_timesteps=total_timesteps)
