import os
import enum
import numpy as np
from gym.envs.mujoco.ant_v3 import AntEnv
from typing import Tuple, NamedTuple, Dict, Optional
from hybrid_gym.model import Mode

mujoco_xml_path: str = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'ant.xml',
))

ROOM_RADIUS: float = 5.0
DOOR_THICKNESS: float = 2.0

class ModeType(enum.Enum):
    STRAIGHT = enum.auto()
    RIGHT = enum.auto()
    LEFT = enum.auto()

class State(NamedTuple):
    qpos: np.ndarray
    qvel: np.ndarray
    observation: np.ndarray
    info: Dict[str, float]
    is_healthy: bool

info_keys: Tuple[str, str, str, str, str, str, str, str, str, str] = (
    'reward_forward',
    'reward_ctrl',
    'reward_contact',
    'reward_survive',
    'x_position',
    'y_position',
    'distance_from_origin',
    'x_velocity',
    'y_velocity',
    'forward_reward',
)

class AntMode(Mode[State]):
    mode_type: ModeType
    ant: AntEnv
    success_bonus: float

    def __init__(self,
                 mode_type: ModeType,
                 xml_file: str = mujoco_xml_path,
                 success_bonus: float = 1e3,
                 **kwargs,
                 ) -> None:
        self.mode_type = mode_type
        self.ant = AntEnv(xml_file=xml_file, **kwargs)
        self.success_bonus = success_bonus
        super().__init__(
            name=mode_type.name,
            action_space=self.ant.action_space,
            observation_space=self.ant.observation_space,
        )

    def set_state(self, st: State) -> None:
        if not np.allclose(
            np.concatenate([st.qpos, st.qvel]),
            self.ant.state_vector(),
        ):
            self.ant.set_state(st.qpos, st.qvel)

    def get_state(self,
                  observation: Optional[np.ndarray] = None,
                  info: Dict[str, float] = dict(),
                  ) -> State:
        xy_array = self.ant.get_body_com('torso')[:2]
        new_info = dict(
            reward_contact=-self.ant.contact_cost,
            reward_survive=self.ant.healthy_reward,
            x_position=xy_array[0],
            y_position=xy_array[1],
            distance_from_origin=np.linalg.norm(xy_array),
        )
        new_info.update(info)
        return State(
            qpos=np.array(self.ant.sim.data.qpos.flat),
            qvel=np.array(self.ant.sim.data.qvel.flat),
            observation=self.ant._get_obs() if observation is None else observation,
            info=new_info,
            is_healthy=self.ant.is_healthy,
        )

    def reset(self) -> State:
        obs = self.ant.reset()
        return self.get_state(observation=obs)

    def is_safe(self, st: State) -> bool:
        wrong_way = (
            st.info['y_position'] < -ROOM_RADIUS - DOOR_THICKNESS
            or (st.info['y_position'] > ROOM_RADIUS + DOOR_THICKNESS
                and self.mode_type != ModeType.STRAIGHT)
            or (st.info['x_position'] < -ROOM_RADIUS - DOOR_THICKNESS
                and self.mode_type != ModeType.LEFT)
            or (st.info['x_position'] > ROOM_RADIUS + DOOR_THICKNESS
                and self.mode_type != ModeType.RIGHT)
        )
        return st.is_healthy and not wrong_way

    def is_success(self, st: State) -> bool:
        return (
            (st.info['y_position'] > ROOM_RADIUS + 0.5 * DOOR_THICKNESS
             and self.mode_type == ModeType.STRAIGHT)
            or (st.info['x_position'] < -ROOM_RADIUS - 0.5 * DOOR_THICKNESS
                and self.mode_type == ModeType.LEFT)
            or (st.info['x_position'] > ROOM_RADIUS + 0.5 * DOOR_THICKNESS
                and self.mode_type == ModeType.RIGHT)
        )

    def render(self, st: State) -> None:
        self.set_state(st)
        self.ant.render()

    def _step_fn(self, st: State, action: np.ndarray) -> State:
        self.set_state(st)
        obs, _, _, info = self.ant.step(action)
        return self.get_state(observation=obs, info=info)

    def _observation_fn(self, st: State) -> np.ndarray:
        return st.observation

    def _reward_fn(self, st0: State, action: np.ndarray, st1: State) -> float:
        if self.mode_type == ModeType.STRAIGHT:
            goal_x = 0.0
            goal_y = ROOM_RADIUS + DOOR_THICKNESS
        elif self.mode_type == ModeType.LEFT:
            goal_x = -ROOM_RADIUS - DOOR_THICKNESS
            goal_y = 0.0
        elif self.mode_type == ModeType.RIGHT:
            goal_x = ROOM_RADIUS + DOOR_THICKNESS
            goal_y = 0.0
        goal_dist0 = np.linalg.norm([
            st0.info['x_position'] - goal_x,
            st0.info['y_position'] - goal_y,
        ])
        goal_dist1 = np.linalg.norm([
            st1.info['x_position'] - goal_x,
            st1.info['y_position'] - goal_y,
        ])
        reward_progress = goal_dist0 - goal_dist1
        return (
            reward_progress + st1.info['reward_survive']
            + st1.info['reward_ctrl'] + st1.info['reward_contact']
            + (self.success_bonus if self.is_success(st1) else 0)
        )

    def vectorize_state(self, st: State) -> np.ndarray:
        return np.concatenate([
            st.qpos, st.qvel, st.observation,
            [st.info.get(k, 0.0) for k in info_keys],
            [1.0 if st.is_healthy else 0.0],
        ])

    def state_from_vector(self, vec: np.ndarray) -> State:
        nq = self.ant.model.nq
        nv = self.ant.model.nv
        nobs, = self.observation_space.size
        return State(
            qpos=vec[:nq], qvel=vec[nq : nq+nv],
            observation=vec[nq+nv : nq+nv+nobs],
            info={
                k: vec[nq + nv + nobs + i]
                for (i, k) in enumerate(info_keys)
            },
            is_healthy=(vec[-1] > 0.5),
        )
