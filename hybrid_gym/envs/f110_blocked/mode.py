import numpy as np
from typing import Optional, Tuple
from hybrid_gym.envs.f110.obstacle_mode import F110ObstacleMode, Polyhedron

def make_blocked(use_throttle: bool,
                 lidar_num_rays: int,
                 wide_width: float,
                 narrow_width: float,
                 length: float,
                 observe_heading: bool,
                 supplement_start_region: bool,
                 observe_previous_lidar: bool,
                 ) -> F110ObstacleMode:
    right_edge = 0.5 * wide_width
    left_edge = -right_edge
    extra_start_region = (
        left_edge + 0.2 * narrow_width,
        right_edge - 0.8 * narrow_width,
        -0.2,
        0.2,
        float(np.radians(60)),
        float(np.radians(120)),
    )
    return F110ObstacleMode(
        name='f110_blocked',
        static_polygons=[],
        static_paths=[
            [
                (left_edge, -length),
                (left_edge, 0.2 * length),
                (right_edge - narrow_width, 0.2 * length),
                (right_edge - narrow_width, 0.3 * length),
                #(left_edge + 0.4 * narrow_width, 0.2 * length),
                #(left_edge + 0.4 * narrow_width, 0.3 * length),
                (left_edge, 0.3 * length),
                (left_edge, 1.2 * length),
                (right_edge - narrow_width, 1.2 * length),
                (right_edge - narrow_width, 1.3 * length),
                #(left_edge + 0.4 * narrow_width, 1.2 * length),
                #(left_edge + 0.4 * narrow_width, 1.3 * length),
                (left_edge, 1.3 * length),
                (left_edge, 1.7 * length),
            ],
            [
                (right_edge, -length),
                #(right_edge, -0.15*length),
                #(left_edge + narrow_width, -0.15 * length),
                #(left_edge + narrow_width, -0.05 * length),
                #(right_edge, -0.05*length),
                #(right_edge, 0.85 * length),
                #(left_edge + narrow_width, 0.85 * length),
                #(left_edge + narrow_width, 0.95 * length),
                #(right_edge, 0.95 * length),
                (right_edge, 1.7 * length),
            ],
        ],
        obstacle_polygons=[],
        obstacle_paths=[],
        obstacle_x_low=-1.0,
        obstacle_y_low=-1.0,
        obstacle_x_high=0.0,
        obstacle_y_high=0.0,
        start_x=right_edge - 0.5 * narrow_width,
        #start_x=0.0,
        start_y=0.0,
        start_V=2.5,
        start_theta=float(np.radians(90)),
        start_x_noise=0.15 * narrow_width,
        #start_x_noise=wide_width / 2 - 0.5,
        start_y_noise=0.2,
        start_V_noise=2.5,
        start_theta_noise=float(np.radians(30)),
        #goal_x=left_edge + 0.5 * narrow_width,
        goal_x=right_edge - 0.5 * narrow_width,
        #goal_x=0.0,
        goal_y=length + 1,
        goal_theta=float(np.radians(90)),
        goal_region=Polyhedron(
            # x >= left_edge
            # x <= right_edge
            # y >= length
            A=np.array([
                [-1, 0, 0, 0],
                [1, 0, 0, 0],
                [0, -1, 0, 0],
            ]),
            b=np.array([-left_edge, right_edge, -length]),
        ),
        center_reward_region=Polyhedron(
            # x >= left_edge
            # x <= right_edge
            # y >= 0.5 * length
            A=np.array([
                [-1, 0, 0, 0],
                [1, 0, 0, 0],
                [0, -1, 0, 0],
            ]),
            b=np.array([-left_edge, right_edge, -0.5 * length]),
        ),
        #center_reward_path=((left_edge + 0.5 * narrow_width, 0), (left_edge + 0.5 * narrow_width, length + 2.0)),
        center_reward_path=((right_edge - 0.5 * narrow_width, 0), (right_edge - 0.5 * narrow_width, length + 2.0)),
        #center_reward_path=((0, 0), (0, length + 2.0)),
        override_start_region=extra_start_region if supplement_start_region else None,
        additional_unsafe_regions=[
            Polyhedron(
                # x >= left_edge
                # x <= right_edge
                # y <= -0.5 * length
                A=np.array([
                    [-1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                ]),
                b=np.array([-left_edge, right_edge, -0.5 * length]),
            ),
        ],
        override_start_transition_pos=(0, 0, float(np.radians(90))),
        override_goal_transition_pos=(0, length, float(np.radians(90))),
        observe_heading=observe_heading,
        mode_onehot_indices=None,
        observe_previous_lidar_avg=observe_previous_lidar,
    )
