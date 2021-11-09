from hybrid_gym.model import Transition
from hybrid_gym.envs.f110.mode import LIDAR_RANGE, State, F110Mode
from hybrid_gym.envs.f110.obstacle_mode import F110ObstacleMode, State as ObstacleState
from typing import Dict, Union
import numpy as np


def straight_length(mode_name: str) -> float:
    length_str = mode_name[len('f110_straight_'):-1]
    if mode_name != 'f110_straight_' + length_str + 'm':
        return 0
    try:
        return float(length_str)
    except ValueError:
        return 0


def flip_sides(mode1: str, mode2: str):
    return ('left' in mode1) != ('left' in mode2)


class F110PlainTrans(Transition):
    plain_modes: Dict[str, F110Mode]
    obstacle_modes: Dict[str, F110ObstacleMode]
    src_is_turn: bool
    hall_width: float

    def __init__(self,
                 source: str,
                 plain_modes: Dict[str, F110Mode],
                 obstacle_modes: Dict[str, F110ObstacleMode],
                 hall_width: float,
                 ) -> None:
        self.plain_modes = plain_modes
        self.obstacle_modes = obstacle_modes
        self.src_is_turn = ('straight' not in source)
        self.hall_width = hall_width
        super().__init__(
            source,
            [m.name for m in plain_modes.values()] +
            [m.name for m in obstacle_modes.values()],
        )

    def guard(self, st: State) -> bool:
        if 'straight' in self.source:
            return st.curHall == 0 and \
                st.car_dist_f <= LIDAR_RANGE + 2
        else:
            return st.curHall == 1

    def jump(self, target: str, st: State) -> Union[State, ObstacleState]:
        dist = straight_length(target)
        new_dist_f = st.car_dist_f - (7 if self.src_is_turn else 0)
        if flip_sides(self.source, target):
            car_dist_s = self.hall_width - st.car_dist_s
        else:
            car_dist_s = st.car_dist_s
        try:
            return self.plain_modes[target].set_state_local(
                car_dist_s, dist + new_dist_f, st.car_heading, st)
        except KeyError:
            m_tgt = self.obstacle_modes[target]
            obstacle_x, obstacle_y = m_tgt.random_obstacle_pos()
            return ObstacleState.make(
                x = m_tgt.start_x + car_dist_s - 0.5 * self.hall_width,
                y = m_tgt.start_y + 5 - new_dist_f,
                V = st.car_V,
                theta = m_tgt.start_theta + st.car_heading,
                obstacle_x = obstacle_x, obstacle_y = obstacle_y,
                lines = m_tgt.compute_obstacle_lines(obstacle_x, obstacle_y),
                num_lidar_rays = m_tgt.num_lidar_rays,
                prev_st = None,
            )

class F110ObstacleTrans(Transition):
    plain_modes: Dict[str, F110Mode]
    obstacle_modes: Dict[str, F110ObstacleMode]

    def __init__(self,
                 source: str,
                 plain_modes: Dict[str, F110Mode],
                 obstacle_modes: Dict[str, F110ObstacleMode],
                 ) -> None:
        self.plain_modes = plain_modes
        self.obstacle_modes = obstacle_modes
        super().__init__(
            source,
            [m.name for m in plain_modes.values()] +
            [m.name for m in obstacle_modes.values()],
        )

    def guard(self, st: ObstacleState) -> bool:
        return self.obstacle_modes[self.source].goal_region.contains(st)

    def jump(self, target: str, st: ObstacleState) -> Union[State, ObstacleState]:
        m_src = self.obstacle_modes[self.source]
        dx = st.x - m_src.goal_x
        dy = st.y - m_src.goal_y
        prev_dx = st.prev_x - m_src.goal_x
        prev_dy = st.prev_y - m_src.goal_y
        try:
            m_tgt = self.obstacle_modes[target]
            change_theta = m_tgt.start_theta - m_src.goal_theta
            rot_matrix = np.array([
                [np.cos(change_theta), -np.sin(change_theta)],
                [np.sin(change_theta), np.cos(change_theta)],
            ])
            src_points = np.array([
                [dx, prev_dx],
                [dy, prev_dy],
            ])
            tgt_points = rot_matrix @ src_points
            tgt_dx = tgt_points[0, 0]
            tgt_dy = tgt_points[1, 0]
            tgt_prev_dx = tgt_points[0, 1]
            tgt_prev_dy = tgt_points[1, 1]
            obstacle_x, obstacle_y = m_tgt.random_obstacle_pos()
            return m_tgt.state_from_scalars(
                x = m_tgt.start_x + tgt_dx,
                y = m_tgt.start_y + tgt_dy,
                V = st.V,
                theta = st.theta + change_theta,
                obstacle_x = obstacle_x, obstacle_y = obstacle_y,
                prev_x = m_tgt.start_x + tgt_prev_dx,
                prev_y = m_tgt.start_y + tgt_prev_dy,
                prev_theta = st.prev_theta + change_theta,
                start_theta = st.theta + change_theta,
            )
        except KeyError:
            plain_tgt = self.plain_modes[target]
            dtheta = st.theta - m_src.goal_theta
            goal_disp_r = np.sqrt(np.square(dx) + np.square(dy))
            goal_disp_theta = np.arctan2(dy, dx)
            proj_dtheta = goal_disp_theta - m_src.goal_theta
            proj_forward = goal_disp_r * np.cos(proj_dtheta)
            proj_right = -goal_disp_r * np.sin(proj_dtheta)
            if flip_sides(self.source, target):
                car_dist_s = 0.5 * plain_tgt.hallWidths[0] - proj_right
            else:
                car_dist_s = 0.5 * plain_tgt.hallWidths[0] + proj_right
            dist = straight_length(target)
            car_dist_f = dist + 7 - proj_forward
            return plain_tgt.set_state_local(
                car_dist_s, car_dist_f, plain_tgt.init_car_heading + dtheta, plain_tgt.reset())
