import numpy as np

from stable_baselines import A2C, ACER, ACKTR, DDPG, DQN, GAIL, HER, PPO1, PPO2, SAC, TD3, TRPO
from stable_baselines.common.base_class import BaseRLModel
from gym.wrappers import TimeLimit
from typing import Iterable, Optional, Callable
from stable_baselines.ddpg.noise import NormalActionNoise
from hybrid_gym.model import Mode, Transition, StateType
from hybrid_gym.envs.pick_place.mode import PickPlaceMode
from hybrid_gym.util.wrappers import GymEnvWrapper, GymGoalEnvWrapper, DoneOnSuccessWrapper


def make_sb_model(mode: Mode,
                  transitions: Iterable[Transition],
                  algo_name: str = 'td3',
                  wrapped_algo: str = 'ddpg',  # only relevent to HER
                  policy: str = 'MlpPolicy',
                  action_noise_scale: float = 0.1,
                  **kwargs
                  ) -> BaseRLModel:
    transition_list = list(transitions)
    env = GymEnvWrapper(mode, transition_list)
    if isinstance(mode, PickPlaceMode):
        goal_env = DoneOnSuccessWrapper(TimeLimit(mode.multi_obj, max_episode_steps=50))
    else:
        goal_env = GymGoalEnvWrapper(mode, transition_list)
    action_shape = mode.action_space.shape
    ddpg_action_noise = NormalActionNoise(
        mean=np.zeros(action_shape),
        sigma=action_noise_scale * np.ones(action_shape)
    )
    if algo_name == 'a2c':
        model: BaseRLModel = A2C(policy, env, **kwargs)
    elif algo_name == 'acer':
        model = ACER(policy, env, **kwargs)
    elif algo_name == 'acktr':
        model = ACKTR(policy, env, **kwargs)
    elif algo_name == 'ddpg':
        model = DDPG(policy, env, action_noise=ddpg_action_noise, **kwargs)
    elif algo_name == 'dqn':
        model = DQN(policy, env, **kwargs)
    elif algo_name == 'gail':
        model = GAIL(policy, env, **kwargs)
    elif algo_name == 'her' and wrapped_algo == 'ddpg':
        model = HER(policy, goal_env, DDPG, action_noise=ddpg_action_noise, **kwargs)
    elif algo_name == 'her' and wrapped_algo == 'dqn':
        model = HER(policy, goal_env, DQN, **kwargs)
    elif algo_name == 'her' and wrapped_algo == 'sac':
        model = HER(policy, goal_env, SAC, **kwargs)
    elif algo_name == 'her' and wrapped_algo == 'td3':
        model = HER(policy, goal_env, TD3, action_noise=ddpg_action_noise, **kwargs)
    elif algo_name == 'ppo1':
        model = PPO1(policy, env, **kwargs)
    elif algo_name == 'ppo2':
        model = PPO2(policy, env, **kwargs)
    elif algo_name == 'sac':
        model = SAC(policy, env, **kwargs)
    elif algo_name == 'td3':
        model = TD3(policy, env, action_noise=ddpg_action_noise, **kwargs)
    elif algo_name == 'trpo':
        model = TRPO(policy, env, **kwargs)
    else:
        raise ValueError
    return model


def train_stable(model, mode: Mode, transitions: Iterable[Transition], total_timesteps=1000,
                 init_states: Optional[Callable[[], StateType]] = None,
                 reward_fn: Optional[Callable[[StateType, np.ndarray, StateType], float]] = None,
                 algo_name='td3'):
    transition_list = list(transitions)
    env = GymEnvWrapper(mode, transition_list, init_states, reward_fn)
    if algo_name == 'her':
        if isinstance(mode, PickPlaceMode):
            env = DoneOnSuccessWrapper(TimeLimit(mode.multi_obj, max_episode_steps=50))
        else:
            env = GymGoalEnvWrapper(mode, transition_list, init_states, reward_fn)
    model.set_env(env)
    model.learn(total_timesteps=total_timesteps)
