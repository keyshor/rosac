import numpy as np
from typing import Dict

from hybrid_gym.model import Transition
from hybrid_gym.envs.f110.obstacle_mode import F110ObstacleMode, State

class F110ObstacleTrans(Transition):
    modes: Dict[str, F110ObstacleMode]
    def __init__(self,
                 source: str,
                 modes: Dict[str, F110ObstacleMode],
                 ) -> None:
        self.modes = modes
        super().__init__(source, [m.name for m in modes.values()])

    def guard(self, st: State) -> bool:
        return self.modes[self.source].goal_region.contains(st)

    def jump(self, target: str, st: State) -> State:
        m_src = self.modes[self.source]
        m_tgt = self.modes[target]
        obstacle_x, obstacle_y = m_tgt.random_obstacle_pos()
        dx = st.x - m_src.goal_x
        dy = st.y - m_src.goal_y
        dtheta = st.theta - m_src.goal_theta
        goal_disp_r = np.sqrt(np.square(dx) + np.square(dy))
        goal_disp_theta = np.arctan2(dy, dx)
        proj_dtheta = goal_disp_theta - m_src.goal_theta
        proj_forward = goal_disp_r * np.cos(proj_dtheta)
        proj_right = -goal_disp_r * np.sin(proj_dtheta)
        return State(
            x = m_tgt.start_x + proj_right,
            y = m_tgt.start_y + proj_forward,
            V = st.V,
            theta = m_tgt.start_theta + dtheta,
            start_theta = m_tgt.start_theta + dtheta,
            obstacle_x = obstacle_x, obstacle_y = obstacle_y,
            lines = m_tgt.compute_obstacle_lines(obstacle_x, obstacle_y),
        )
