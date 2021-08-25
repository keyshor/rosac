import numpy as np
import os

from stable_baselines import A2C, ACER, ACKTR, DDPG, DQN, GAIL, HER, PPO1, PPO2, SAC, TD3, TRPO
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.policies import BasePolicy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.her import HERGoalEnvWrapper
import gym
from gym.wrappers import TimeLimit
from typing import Iterable, Optional, List, Tuple, Union, Callable, TypeVar, Type, Any
from stable_baselines.ddpg.noise import NormalActionNoise

from spectrl.rl.ddpg import DDPG as SpectrlDdpg, DDPGParams as SpectrlDdpgParams

from hybrid_gym.model import Mode, Transition, StateType
from hybrid_gym.util.wrappers import (
    GymMultiEnvWrapper, GymMultiGoalEnvWrapper,
    DoneOnSuccessWrapper, BaselineCtrlWrapper,
    GymEnvWrapper,
)


BasePolicySubclass = TypeVar('BasePolicySubclass', bound=BasePolicy)


def env_from_mode_info(raw_mode_info: Iterable[Tuple[
    Mode[StateType],
    Iterable[Transition],
    Optional[Callable[[], StateType]],
    Optional[Callable[[StateType, np.ndarray, StateType], float]],
]],
    algo_name: str,
    max_episode_steps: int,
) -> gym.Env:
    mode_info = [
        (mode, list(transitions), reset_fn, reward_fn)
        for (mode, transitions, reset_fn, reward_fn) in raw_mode_info
    ]
    if algo_name == 'td3':
        return TimeLimit(GymMultiEnvWrapper(mode_info, flatten_obs=True),
                         max_episode_steps=max_episode_steps)
    elif algo_name == 'her':
        return HERGoalEnvWrapper(DoneOnSuccessWrapper(TimeLimit(
            GymMultiGoalEnvWrapper(mode_info),
            max_episode_steps=max_episode_steps,
        )))
    else:
        return TimeLimit(GymMultiEnvWrapper(mode_info),
                         max_episode_steps=max_episode_steps)


def make_spectrl_model(modes: Iterable[Mode[StateType]],
                       max_episode_steps: int = 50,
                       **kwargs,
                       ) -> SpectrlDdpg:
    mode_info: List[Tuple[
        Mode[StateType],
        List[Transition],
        Optional[Callable[[], StateType]],
        Optional[Callable[[StateType, np.ndarray, StateType], float]],
    ]] = [(m, [], None, None) for m in modes]
    env = TimeLimit(GymMultiEnvWrapper(mode_info, flatten_obs=True),
                    max_episode_steps=max_episode_steps)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high
    params = SpectrlDdpgParams(state_dim, action_dim, action_bound, **kwargs)
    model = SpectrlDdpg(params)
    return model


def train_spectrl(model,
                  mode_info: Iterable[Tuple[
                      Mode[StateType],
                      Iterable[Transition],
                      Optional[Callable[[], StateType]],
                      Optional[Callable[[StateType, np.ndarray, StateType], float]],
                  ]],
                  max_episode_steps: int = 50,
                  ) -> None:
    env = TimeLimit(GymMultiEnvWrapper(mode_info, flatten_obs=True),
                    max_episode_steps=max_episode_steps)
    model.train(env)


def make_sb_model(raw_mode_info: Iterable[Tuple[
    Mode[StateType],
    Iterable[Transition],
    Optional[Callable[[], StateType]],
    Optional[Callable[[StateType, np.ndarray, StateType], float]],
]],
    algo_name: str = 'td3',
    wrapped_algo: str = 'ddpg',  # only relevent to HER
    action_noise_scale: float = 0.1,
    policy: Union[Type[BasePolicySubclass], str] = 'MlpPolicy',
    max_episode_steps: int = 50,
    **kwargs
) -> BaseRLModel:
    mode_info = [
        (mode, list(transitions), reset_fn, reward_fn)
        for (mode, transitions, reset_fn, reward_fn) in raw_mode_info
    ]
    env = env_from_mode_info(mode_info, algo_name, max_episode_steps)
    action_shape = env.action_space.shape
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
    elif algo_name == 'her':
        if wrapped_algo == 'ddpg':
            model = HER(policy, env, DDPG, action_noise=ddpg_action_noise, **kwargs)
        elif wrapped_algo == 'dqn':
            model = HER(policy, env, DQN, **kwargs)
        elif wrapped_algo == 'sac':
            model = HER(policy, env, SAC, **kwargs)
        elif wrapped_algo == 'td3':
            model = HER(policy, env, TD3, action_noise=ddpg_action_noise, **kwargs)
        else:
            raise ValueError
    elif algo_name == 'ppo1':
        model = PPO1(policy, env, **kwargs)
    elif algo_name == 'ppo2':
        model = PPO2(policy, env, **kwargs)
    elif algo_name == 'sac':
        model = SAC(policy, env, **kwargs)
    elif algo_name == 'td3':
        # env = GymMultiEnvWrapper(mode, transition_list, flatten_obs=True)
        model = TD3(policy, env, action_noise=ddpg_action_noise, **kwargs)
    elif algo_name == 'trpo':
        model = TRPO(policy, env, **kwargs)
    else:
        raise ValueError
    return model


def check_initialization(model: BaseRLModel,
                         env: gym.Env,
                         n_eval_episodes: int,
                         num_timesteps: int,
                         min_reward: float,
                         min_episode_length: float,
                         ) -> bool:
    model.set_env(env)
    model.learn(total_timesteps=num_timesteps)
    episode_reward, episode_timesteps = evaluate_policy(
        model, env, n_eval_episodes=n_eval_episodes,
        return_episode_rewards=True,
    )
    return np.mean(episode_reward) >= min_reward and \
        np.mean(episode_timesteps) >= min_episode_length


def make_sb_model_init_check(raw_mode_info: Iterable[Tuple[
    Mode[StateType],
    Iterable[Transition],
    Optional[Callable[[], StateType]],
    Optional[Callable[[StateType, np.ndarray, StateType], float]],
]],
    save_path: str = '.',
    algo_name: str = 'td3',
    wrapped_algo: str = 'ddpg',  # only relevent to HER
    action_noise_scale: float = 0.1,
    policy: Union[Type[BasePolicySubclass], str] = 'MlpPolicy',
    max_episode_steps: int = 50,
    max_init_retries: int = 10,
    n_eval_episodes: int = 100,
    num_timesteps: int = 1000,
    min_reward: float = -np.inf,
    min_episode_length: float = 0.0,
    **kwargs,
) -> BaseRLModel:
    mode_info = [
        (mode, list(transitions), reset_fn, reward_fn)
        for (mode, transitions, reset_fn, reward_fn) in raw_mode_info
    ]
    first_mode, _, _, _ = mode_info[0]
    env = env_from_mode_info(mode_info, algo_name=algo_name, max_episode_steps=max_episode_steps)
    init_ok = False
    for i in range(max_init_retries):
        model = make_sb_model(
            mode_info,
            algo_name=algo_name,
            policy=policy,
            action_noise_scale=action_noise_scale,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )
        init_ok = check_initialization(
            model, env,
            n_eval_episodes=n_eval_episodes,
            num_timesteps=num_timesteps,
            min_reward=min_reward,
            min_episode_length=min_episode_length,
        )
        if init_ok:
            print(f'initialized model after {i+1} attempts')
            break
    if not init_ok:
        print(f'failed to achieve suitable initialization after {max_init_retries}')
    ctrl = BaselineCtrlWrapper(model)
    ctrl.save(os.path.join(save_path, f'{first_mode.name}.{algo_name}'))
    ctrl = BaselineCtrlWrapper.load(os.path.join(save_path, f'{first_mode.name}.{algo_name}'),
                                    algo_name=algo_name, env=env)
    return ctrl.model


def train_stable(model,
                 raw_mode_info: Iterable[Tuple[
                     Mode[StateType],
                     Iterable[Transition],
                     Optional[Callable[[], StateType]],
                     Optional[Callable[[StateType, np.ndarray, StateType], float]],
                 ]],
                 total_timesteps=1000,
                 algo_name: str = 'td3',
                 max_episode_steps: int = 50,
                 eval_freq: int = 10000,
                 n_eval_episodes: int = 100,
                 save_path: str = '.',
                 custom_best_model_path: Optional[str] = None,
                 ) -> Any:
    mode_info = [
        (mode, list(transitions), reset_fn, reward_fn)
        for (mode, transitions, reset_fn, reward_fn) in raw_mode_info
    ]
    env = env_from_mode_info(mode_info, algo_name=algo_name, max_episode_steps=max_episode_steps)
    model.set_env(env)
    first_mode, _, _, _ = mode_info[0]
    best_model_path = custom_best_model_path or first_mode.name
    callback = EvalCallback(
        eval_env=env, n_eval_episodes=n_eval_episodes, eval_freq=eval_freq,
        log_path=os.path.join(save_path, best_model_path),
        best_model_save_path=os.path.join(save_path, best_model_path),
    )
    model.learn(total_timesteps=total_timesteps, callback=callback)
    return env
