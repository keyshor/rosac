import numpy as np

from stable_baselines import A2C, ACER, ACKTR, DDPG, DQN, GAIL, HER, PPO1, PPO2, SAC, TD3, TRPO
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.her import HERGoalEnvWrapper
from gym.wrappers import TimeLimit
from typing import Iterable, Optional, Callable, Any
from stable_baselines.ddpg.noise import NormalActionNoise

from spectrl.rl.ddpg import DDPG as SpectrlDdpg, DDPGParams as SpectrlDdpgParams

from hybrid_gym.model import Mode, Transition, StateType
from hybrid_gym.util.wrappers import GymEnvWrapper, GymGoalEnvWrapper, DoneOnSuccessWrapper

def make_spectrl_model(mode: Mode,
                       transitions: Iterable[Transition],
                       max_episode_steps: int = 50,
                       **kwargs,
                       ) -> SpectrlDdpg:
    transition_list = list(transitions)
    env = TimeLimit(GymEnvWrapper(mode, transition_list, flatten_obs=True), max_episode_steps=max_episode_steps)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high
    params = SpectrlDdpgParams(state_dim, action_dim, action_bound, **kwargs)
    model = SpectrlDdpg(params)
    return model

def train_spectrl(model, mode: Mode, transitions: Iterable[Transition],
                  init_states: Optional[Callable[[], StateType]] = None,
                  reward_fn: Optional[Callable[[StateType, np.ndarray, StateType], float]] = None,
                  max_episode_steps: int = 50,
                  ) -> None:
    transition_list = list(transitions)
    env = TimeLimit(GymEnvWrapper(mode, transition_list, init_states, reward_fn, flatten_obs=True), max_episode_steps=max_episode_steps)
    model.train(env)

def make_sb_model(mode: Mode,
                  transitions: Iterable[Transition],
                  algo_name: str = 'td3',
                  wrapped_algo: str = 'ddpg',  # only relevent to HER
                  action_noise_scale: float = 0.1,
                  custom_policy = None,
                  max_episode_steps: int = 50,
                  **kwargs
                  ) -> BaseRLModel:
    transition_list = list(transitions)
    env = TimeLimit(GymEnvWrapper(mode, transition_list), max_episode_steps=max_episode_steps)
    goal_env = HERGoalEnvWrapper(DoneOnSuccessWrapper(TimeLimit(
        GymGoalEnvWrapper(mode, transition_list),
        max_episode_steps=max_episode_steps,
    )))
    action_shape = mode.action_space.shape
    ddpg_action_noise = NormalActionNoise(
        mean=np.zeros(action_shape),
        sigma=action_noise_scale * np.ones(action_shape)
    )
    policy: Any = custom_policy or 'MlpPolicy'
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
        #env = GymEnvWrapper(mode, transition_list, flatten_obs=True)
        model = TD3(policy, env, action_noise=ddpg_action_noise, **kwargs)
    elif algo_name == 'trpo':
        model = TRPO(policy, env, **kwargs)
    else:
        raise ValueError
    return model


def train_stable(model, mode: Mode, transitions: Iterable[Transition], total_timesteps=1000,
                 init_states: Optional[Callable[[], StateType]] = None,
                 reward_fn: Optional[Callable[[StateType, np.ndarray, StateType], float]] = None,
                 algo_name='td3',
                 max_episode_steps: int = 50,
                 ) -> None:
    transition_list = list(transitions)
    if algo_name == 'td3':
        env = TimeLimit(GymEnvWrapper(mode, transition_list, init_states, reward_fn, flatten_obs=True), max_episode_steps=max_episode_steps)
    elif algo_name == 'her':
        env = HERGoalEnvWrapper(DoneOnSuccessWrapper(TimeLimit(
            GymGoalEnvWrapper(mode, transition_list, init_states, reward_fn),
            max_episode_steps=max_episode_steps,
        )))
    else:
        env = TimeLimit(GymEnvWrapper(mode, transition_list, init_states, reward_fn), max_episode_steps=max_episode_steps)
    model.set_env(env)
    callback = EvalCallback(
        eval_env=env, n_eval_episodes=100, eval_freq=10000,
        #log_path=f'{mode.name}', best_model_save_path=f'{mode.name}',
    )
    model.learn(total_timesteps=total_timesteps, callback=callback)
