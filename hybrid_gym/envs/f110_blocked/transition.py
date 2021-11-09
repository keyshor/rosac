from hybrid_gym.model import Transition
from hybrid_gym.envs.f110.obstacle_mode import F110ObstacleMode, State, LIDAR_RANGE
from typing import Iterable, Dict
import numpy as np

class F110BlockedTrans(Transition):
    modes: Dict[str, F110ObstacleMode]

    def __init__(self,
                 source: str,
                 modes: Iterable[F110ObstacleMode],
                 ) -> None:
        self.modes = {m.name: m for m in modes}
        super().__init__(
            source,
            [m_name for m_name in self.modes.keys()],
        )

    def guard(self, st: State) -> bool:
        return self.modes[self.source].goal_region.contains(st)

    def jump(self, target: str, st: State) -> State:
        m_src = self.modes[self.source]
        dx = st.x - m_src.goal_trans_x
        dy = st.y - m_src.goal_trans_y
        prev_dx = st.prev_x - m_src.goal_trans_x
        prev_dy = st.prev_y - m_src.goal_trans_y
        m_tgt = self.modes[target]
        change_theta = m_tgt.start_trans_theta - m_src.goal_trans_theta
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
            x = m_tgt.start_trans_x + tgt_dx,
            y = m_tgt.start_trans_y + tgt_dy,
            V = st.V,
            theta = st.theta + change_theta,
            obstacle_x = obstacle_x, obstacle_y = obstacle_y,
            prev_x = m_tgt.start_trans_x + tgt_prev_dx,
            prev_y = m_tgt.start_trans_y + tgt_prev_dy,
            prev_theta = st.prev_theta + change_theta,
            start_theta = st.theta + change_theta,
        )
