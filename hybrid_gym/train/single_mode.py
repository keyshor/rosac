import numpy as np
import os

from stable_baselines import A2C, ACER, ACKTR, DDPG, DQN, GAIL, HER, PPO1, PPO2, SAC, TD3, TRPO
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.policies import BasePolicy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.ddpg.noise import NormalActionNoise
from stable_baselines3 import (
    A2C as SB3_A2C, DDPG as SB3_DDPG, DQN as SB3_DQN,
    PPO as SB3_PPO, SAC as SB3_SAC, TD3 as SB3_TD3
)
from sb3_contrib import TQC as SB3_TQC, QRDQN as SB3_QRDQN
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback as SB3_EvalCallback
from stable_baselines3.common.policies import BasePolicy as SB3_BasePolicy, ActorCriticPolicy
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.td3.policies import TD3Policy
from sb3_contrib.tqc.policies import TQCPolicy
from sb3_contrib.qrdqn.policies import QRDQNPolicy
from stable_baselines3.common.evaluation import evaluate_policy as sb3_evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise as SB3_NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import TimeFeatureWrapper
import gym
from gym.wrappers import TimeLimit
from typing import Iterable, Optional, List, Tuple, Union, Callable, Type, Any

from spectrl.rl.ddpg import DDPG as SpectrlDdpg, DDPGParams as SpectrlDdpgParams

from hybrid_gym.model import Mode, Transition, StateType
from hybrid_gym.util.wrappers import (
    GymMultiEnvWrapper, GymMultiGoalEnvWrapper, BaselineCtrlWrapper, GymEnvWrapper
)

from hybrid_gym.rl.ars import NNParams, ARSParams, ARSModel


# BasePolicySubclass = TypeVar('BasePolicySubclass', bound=BasePolicy)
def env_from_mode_info(
        raw_mode_info: Iterable[Tuple[
            Mode[StateType],
            Iterable[Transition],
            Optional[Callable[[], StateType]],
            Optional[Callable[[StateType, np.ndarray, StateType], float]],
        ]],
        algo_name: str,
        max_episode_steps: int,
        reward_offset: float,
        is_goal_env: bool) -> gym.Env:
    mode_info = [
        (mode, list(transitions), reset_fn, reward_fn)
        for (mode, transitions, reset_fn, reward_fn) in raw_mode_info
    ]
    if is_goal_env:
        return TimeLimit(
            GymMultiGoalEnvWrapper(mode_info),
            max_episode_steps=max_episode_steps,
        )
    else:
        return TimeLimit(GymMultiEnvWrapper(mode_info, flatten_obs=True),
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


def make_sb_model(
        raw_mode_info: Iterable[Tuple[
            Mode[StateType],
            Iterable[Transition],
            Optional[Callable[[], StateType]],
            Optional[Callable[[StateType, np.ndarray, StateType], float]],
        ]],
        algo_name: str = 'td3',
        wrapped_algo: str = 'ddpg',  # only relevent to HER
        action_noise_scale: float = 0.1,
        policy: Union[Type[BasePolicy], str] = 'MlpPolicy',
        max_episode_steps: int = 50,
        reward_offset: float = 1.0,
        is_goal_env: bool = False,
        **kwargs) -> BaseRLModel:
    mode_info = [
        (mode, list(transitions), reset_fn, reward_fn)
        for (mode, transitions, reset_fn, reward_fn) in raw_mode_info
    ]
    env = env_from_mode_info(mode_info, algo_name, max_episode_steps,
                             reward_offset, is_goal_env=is_goal_env)
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


def make_sb_model_init_check(
        raw_mode_info: Iterable[Tuple[
            Mode[StateType],
            Iterable[Transition],
            Optional[Callable[[], StateType]],
            Optional[Callable[[StateType, np.ndarray, StateType], float]],
        ]],
        save_path: str = '.',
        algo_name: str = 'td3',
        wrapped_algo: str = 'ddpg',  # only relevent to HER
        action_noise_scale: float = 0.1,
        policy: Union[Type[BasePolicy], str] = 'MlpPolicy',
        max_episode_steps: int = 50,
        max_init_retries: int = 10,
        n_eval_episodes: int = 100,
        num_timesteps: int = 1000,
        min_reward: float = -np.inf,
        min_episode_length: float = 0.0,
        reward_offset: float = 1.0,
        is_goal_env: bool = False,
        **kwargs,) -> BaseRLModel:
    mode_info = [
        (mode, list(transitions), reset_fn, reward_fn)
        for (mode, transitions, reset_fn, reward_fn) in raw_mode_info
    ]
    first_mode, _, _, _ = mode_info[0]
    env = env_from_mode_info(mode_info, algo_name=algo_name, max_episode_steps=max_episode_steps,
                             reward_offset=reward_offset, is_goal_env=is_goal_env)
    init_ok = False
    for i in range(max_init_retries):
        model = make_sb_model(
            mode_info,
            algo_name=algo_name,
            wrapped_algo=wrapped_algo,
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
                 reward_offset: float = 1.0,
                 save_path: str = '.',
                 custom_best_model_path: Optional[str] = None,
                 is_goal_env: bool = False,
                 ) -> Any:
    mode_info = [
        (mode, list(transitions), reset_fn, reward_fn)
        for (mode, transitions, reset_fn, reward_fn) in raw_mode_info
    ]
    env = env_from_mode_info(mode_info, algo_name=algo_name, max_episode_steps=max_episode_steps,
                             reward_offset=reward_offset, is_goal_env=is_goal_env)
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


def make_sb3_model(
        raw_mode_info: Iterable[Tuple[
            Mode[StateType],
            Iterable[Transition],
            Optional[Callable[[], StateType]],
            Optional[Callable[[StateType, np.ndarray, StateType], float]],
        ]],
        algo_name: str,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        action_noise_scale: float = 0.1,
        policy: Union[Type[SB3_BasePolicy], str] = 'MlpPolicy',
        max_episode_steps: int = 50,
        reward_offset: float = 1.0,
        is_goal_env: bool = False,
        **kwargs) -> BaseAlgorithm:
    mode_info = [
        (mode, list(transitions), reset_fn, reward_fn)
        for (mode, transitions, reset_fn, reward_fn) in raw_mode_info
    ]
    env = env_from_mode_info(mode_info, algo_name=algo_name, max_episode_steps=max_episode_steps,
                             reward_offset=reward_offset, is_goal_env=is_goal_env)
    action_shape = env.action_space.shape
    action_noise = SB3_NormalActionNoise(
        mean=np.zeros(action_shape),
        sigma=action_noise_scale * np.ones(action_shape)
    )
    if algo_name == 'a2c':
        assert isinstance(policy, str) or issubclass(policy, ActorCriticPolicy), \
            'policies passed to A2C must be subclasses of ActorCriticPolicy'
        model: BaseAlgorithm = SB3_A2C(policy, env, **kwargs)
    elif algo_name == 'ddpg':
        assert isinstance(policy, str) or issubclass(policy, TD3Policy), \
            'policies passed to DDPG must be subclasses of TD3Policy'
        model = SB3_DDPG(policy, env, action_noise=action_noise,
                         replay_buffer_class=replay_buffer_class, **kwargs)
    elif algo_name == 'dqn':
        assert isinstance(policy, str) or issubclass(policy, DQNPolicy), \
            'policies passed to DQN must be subclasses of DQNPolicy'
        model = SB3_DQN(policy, env, replay_buffer_class=replay_buffer_class, **kwargs)
    elif algo_name == 'ppo':
        assert isinstance(policy, str) or issubclass(policy, ActorCriticPolicy), \
            'policies passed to PPO must be subclasses of ActorCriticPolicy'
        model = SB3_PPO(policy, env, **kwargs)
    elif algo_name == 'sac':
        assert isinstance(policy, str) or issubclass(policy, SACPolicy), \
            'policies passed to SAC must be subclasses of SACPolicy'
        model = SB3_SAC(policy, env, replay_buffer_class=replay_buffer_class, **kwargs)
    elif algo_name == 'td3':
        assert isinstance(policy, str) or issubclass(policy, TD3Policy), \
            'policies passed to TD3 must be subclasses of TD3Policy'
        model = SB3_TD3(policy, env, action_noise=action_noise,
                        replay_buffer_class=replay_buffer_class, **kwargs)
    elif algo_name == 'tqc':
        assert isinstance(policy, str) or issubclass(policy, TQCPolicy), \
            'policies passed to TQC must be subclasses of TQCPolicy'
        model = SB3_TQC(policy, env, action_noise=action_noise,
                        replay_buffer_class=replay_buffer_class, **kwargs)
    elif algo_name == 'qrdqn':
        assert isinstance(policy, str) or issubclass(policy, QRDQNPolicy), \
            'policies passed to QRDQN must be subclasses of QRDQNPolicy'
        model = SB3_QRDQN(policy, env,
                        replay_buffer_class=replay_buffer_class, **kwargs)
    else:
        raise ValueError
    return model


def sb3_check_initialization(model: BaseAlgorithm,
                             env: gym.Env,
                             n_eval_episodes: int,
                             num_timesteps: int,
                             min_reward: float,
                             min_episode_length: float,
                             ) -> bool:
    model.set_env(env)
    model.learn(total_timesteps=num_timesteps)
    episode_reward, episode_timesteps = sb3_evaluate_policy(
        model, env, n_eval_episodes=n_eval_episodes,
        return_episode_rewards=True,
    )
    return np.mean(episode_reward) >= min_reward and \
        np.mean(episode_timesteps) >= min_episode_length


def make_sb3_model_init_check(
        raw_mode_info: Iterable[Tuple[
            Mode[StateType],
            Iterable[Transition],
            Optional[Callable[[], StateType]],
            Optional[Callable[[StateType, np.ndarray, StateType], float]],
        ]],
        algo_name: str = 'td3',
        replay_buffer_class: Optional[ReplayBuffer] = None,
        action_noise_scale: float = 0.1,
        policy: Union[Type[SB3_BasePolicy], str] = 'MlpPolicy',
        max_episode_steps: int = 50,
        max_init_retries: int = 10,
        n_eval_episodes: int = 100,
        num_timesteps: int = 1000,
        min_reward: float = -np.inf,
        min_episode_length: float = 0.0,
        reward_offset: float = 1.0,
        is_goal_env: bool = False,
        **kwargs) -> BaseAlgorithm:
    mode_info = [
        (mode, list(transitions), reset_fn, reward_fn)
        for (mode, transitions, reset_fn, reward_fn) in raw_mode_info
    ]
    first_mode, _, _, _ = mode_info[0]
    env = env_from_mode_info(mode_info, algo_name=algo_name, max_episode_steps=max_episode_steps,
                             reward_offset=reward_offset, is_goal_env=is_goal_env)
    init_ok = False
    for i in range(max_init_retries):
        model = make_sb3_model(
            mode_info,
            algo_name=algo_name,
            policy=policy,
            action_noise_scale=action_noise_scale,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )
        init_ok = sb3_check_initialization(
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
    return model


def train_sb3(model: BaseAlgorithm,
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
              reward_offset: float = 1.0,
              save_path: str = '.',
              custom_best_model_path: Optional[str] = None,
              is_goal_env: bool = False,
              **kwargs) -> Any:
    mode_info = [
        (mode, list(transitions), reset_fn, reward_fn)
        for (mode, transitions, reset_fn, reward_fn) in raw_mode_info
    ]
    env = env_from_mode_info(mode_info, algo_name=algo_name, max_episode_steps=max_episode_steps,
                             reward_offset=reward_offset, is_goal_env=is_goal_env)
    model.set_env(env)
    first_mode, _, _, _ = mode_info[0]
    best_model_path = custom_best_model_path or first_mode.name
    callback = SB3_EvalCallback(
        eval_env=Monitor(env), n_eval_episodes=n_eval_episodes, eval_freq=eval_freq,
        log_path=os.path.join(save_path, best_model_path),
        best_model_save_path=os.path.join(save_path, best_model_path),
    )
    model.learn(total_timesteps=total_timesteps, callback=callback)
    return env


def make_ars_model(
        ars_params: ARSParams,
        nn_params: NNParams,
        use_gpu: bool = False,
        **kwargs) -> ARSModel:
    return ARSModel(nn_params, ars_params, use_gpu=use_gpu)


def learn_ars_model(model: ARSModel,
                    raw_mode_info: List[Tuple[
                        Mode[StateType],
                        Iterable[Transition],
                        Optional[Callable[[], StateType]],
                        Optional[Callable[[StateType, np.ndarray, StateType], float]],
                    ]],
                    save_path: str = '.',
                    custom_best_model_path: Optional[str] = None,
                    verbose=False) -> None:
    env_list = [GymEnvWrapper(*mode_info) for mode_info in raw_mode_info]
    best_policy, _ = model.learn(env_list, verbose=verbose)

    first_mode, _, _, _ = raw_mode_info[0]
    best_model_path = custom_best_model_path or first_mode.name
    best_policy.save(best_model_path, save_path)


def parallel_ars(model, mode_info, save_path, ret_queue, req_queue, verbose, use_gpu):
    if use_gpu:
        model.gpu()
    learn_ars_model(model, mode_info, save_path=save_path, verbose=verbose)
    while req_queue.get() is not None:
        if use_gpu:
            model.cpu()
        ret_queue.put(model)
