import gym
import numpy as np
from typing import Any


def obs_shape(obs: Any) -> Any:
    if isinstance(obs, np.ndarray):
        return obs.shape
    elif isinstance(obs, dict):
        return {k: obs_shape(v) for (k, v) in obs.items()}
    else:
        return 'unknown_obs'


def space_shape(sp: gym.Space) -> Any:
    if isinstance(sp, gym.spaces.Box):
        return sp.shape
    elif isinstance(sp, gym.spaces.Dict):
        return {k: space_shape(v) for (k, v) in sp.spaces.items()}
    else:
        return 'unknown_space'


def obs_dtype(obs: Any) -> Any:
    if isinstance(obs, np.ndarray):
        return obs.dtype
    elif isinstance(obs, dict):
        return {k: obs_dtype(v) for (k, v) in obs.items()}
    else:
        return 'unknown_obs'


def space_dtype(sp: gym.Space) -> Any:
    if isinstance(sp, gym.spaces.Box):
        return sp.dtype
    elif isinstance(sp, gym.spaces.Dict):
        return {k: space_dtype(v) for (k, v) in sp.spaces.items()}
    else:
        return 'unknown_space'
