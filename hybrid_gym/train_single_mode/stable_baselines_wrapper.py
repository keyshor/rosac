import gym
import numpy as np
from stable_baselines import A2C, ACER, ACKTR, DDPG, DQN, GAIL, HER, PPO1, PPO2, SAC, TD3, TRPO
from stable_baselines.common.policies import BasePolicy
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.policies import MlpPolicy as MlpCommon
from stable_baselines.ddpg.policies import MlpPolicy as MlpDdpg
from stable_baselines.deepq.policies import MlpPolicy as MlpDqn
from stable_baselines.sac.policies import MlpPolicy as MlpSac
from stable_baselines.td3.policies import MlpPolicy as MlpTd3
from typing import Iterable, List, Tuple, Dict, Optional, Union, Callable, NoReturn, Generic, TypeVar
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

    def reset(self) -> np.ndarray:
        self.state = self.reset_fn()
        return self.mode.observe(self.state)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        next_state = self.mode.step(self.state, action)
        reward = self.reward_fn(self.state, action, next_state)
        self.state = next_state
        done = False
        for t in self.transitions:
            if t.guard(self.state):
                done = True
                break
        return self.mode.observe(self.state), reward, done, {}

    def render(self) -> None:
        self.mode.render(self.state)

def BaselineCtrlWrapper(Controller):
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

def train_stable(mode: Mode[StateType],
                 transitions: Iterable[Transition],
                 algo_name: str = 'td3',
                 wrapped_algo: str = 'ddpg', # only relevent to HER
                 policy: Optional[BasePolicy] = None,
                 total_timesteps: int = 10000,
                 init_states: Optional[Callable[[], StateType]] = None,
                 reward_fn: Optional[Callable[[StateType, np.ndarray, StateType], float]] = None,
                 verbose: int = 0
                 ) -> Controller:
    env = GymEnvWrapper(mode, transitions, init_states, reward_fn)
    common_policy = policy or MlpCommon
    if algo_name == 'a2c':
        model: BaseRLModel = A2C(common_policy, env, verbose=verbose)
    elif algo_name == 'acer':
        model = ACER(common_policy, env, verbose=verbose)
    elif algo_name == 'acktr':
        model = ACKTR(common_policy, env, verbose=verbose)
    elif algo_name == 'ddpg':
        model = DDPG(policy or MlpDdpg, env, verbose=verbose)
    elif algo_name == 'dqn':
        model = DQN(policy or MlpDqn, env, verbose=verbose)
    elif algo_name == 'gail':
        model = GAIL(common_policy, env, verbose=verbose)
    elif algo_name == 'her' and wrapped_algo == 'ddpg':
        model = HER(common_policy, env, DDPG, verbose=verbose)
    elif algo_name == 'her' and wrapped_algo == 'dqn':
        model = HER(common_policy, env, DQN, verbose=verbose)
    elif algo_name == 'her' and wrapped_algo == 'sac':
        model = HER(common_policy, env, SAC, verbose=verbose)
    elif algo_name == 'her' and wrapped_algo == 'td3':
        model = HER(common_policy, env, TD3, verbose=verbose)
    elif algo_name == 'ppo1':
        model = PPO1(common_policy, env, verbose=verbose)
    elif algo_name == 'ppo2':
        model = PPO2(common_policy, env, verbose=verbose)
    elif algo_name == 'sac':
        model = SAC(policy or MlpSac, env, verbose=verbose)
    elif algo_name == 'td3':
        model = TD3(policy or MlpTd3, env, verbose=verbose)
    elif algo_name == 'trpo':
        model = TRPO(common_policy, env, verbose=verbose)
    else:
        raise ValueError
    model.learn(total_timesteps=total_timesteps)
    ctrl = BaselineCtrlWrapper(model)
    return ctrl
