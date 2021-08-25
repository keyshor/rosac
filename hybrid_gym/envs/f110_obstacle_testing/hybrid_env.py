import numpy as np
from hybrid_gym.hybrid_env import HybridAutomaton
from hybrid_gym.envs.f110_obstacle.mode import F110ObstacleMode, Polyhedron
from hybrid_gym.envs.f110_obstacle.transition import F110ObstacleTrans
from typing import Iterable

def make_f110_model(straight_lengths: Iterable[float] = [10.0],
                    num_lidar_rays: int = 1081,
                    hall_width: float = 1.5,
                    ) -> HybridAutomaton:
    hhw = hall_width / 2
    straight_modes = [
        F110ObstacleMode(
            name=f'f110_straight_{l}m',
            static_polygons=[],
            static_paths=[
                [(-hhw,-10), (-hhw, l + 10)],
                [(hhw,-10), (hhw, l + 10)],
            ],
            obstacle_polygons=[],
            obstacle_paths=[],
            obstacle_x_low=0,
            obstacle_x_high=0,
            obstacle_y_low=0,
            obstacle_y_high=0,
            start_x=0,
            start_y=0,
            start_V = 2.5,
            start_theta=float(np.radians(90)),
            start_x_noise = 0.2,
            start_y_noise = 0.2,
            start_V_noise = 2.5,
            start_theta_noise = float(np.radians(30)),
            goal_x=0,
            goal_y=l,
            goal_theta=float(np.radians(90)),
            goal_region=Polyhedron(
                # y >= l
                # x >= -hhw
                # x <= hhw
                A=np.array([
                    [0, -1, 0, 0],
                    [-1, 0, 0, 0],
                    [1, 0, 0, 0],
                ]),
                b=np.array([-l, hhw, hhw]),
            ),
            center_reward_region=Polyhedron(
                # y >= -5
                # x >= -hhw
                # x <= hhw
                A=np.array([
                    [0, -1, 0, 0],
                    [-1, 0, 0, 0],
                    [1, 0, 0, 0],
                ]),
                b=np.array([5, hhw, hhw]),
            ),
            center_reward_path=((0, -5), (0, l+5)),
        )
        for l in straight_lengths
    ]
    square_right = F110ObstacleMode(
        name='f110_square_right',
        static_polygons=[],
        static_paths=[
            [(-hhw,-10), (-hhw,hhw), (10,hhw)],
            [(hhw,-10), (hhw,-hhw), (10,-hhw)],
        ],
        obstacle_polygons=[],
        obstacle_paths=[],
        obstacle_x_low=0,
        obstacle_x_high=0,
        obstacle_y_low=0,
        obstacle_y_high=0,
        start_x=0,
        start_y=-7,
        start_V = 2.5,
        start_theta=float(np.radians(90)),
        start_x_noise = 0.2,
        start_y_noise = 0.2,
        start_V_noise = 2.5,
        start_theta_noise = float(np.radians(30)),
        goal_x=7,
        goal_y=0,
        goal_theta=0,
        goal_region=Polyhedron(
            # x >= 7
            # y >= -hhw
            # y <= hhw
            A=np.array([
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 1, 0, 0],
            ]),
            b=np.array([-7, hhw, hhw]),
        ),
        center_reward_region=Polyhedron(
            # x >= hhw
            # y >= -hhw
            # y <= hhw
            A=np.array([
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 1, 0, 0],
            ]),
            b=np.array([-hhw, hhw, hhw]),
        ),
        center_reward_path=((0, 0), (10, 0)),
    )
    square_left = F110ObstacleMode(
        name='f110_square_left',
        static_polygons=[],
        static_paths=[
            [(-hhw,-10), (-hhw,-hhw), (-10,-hhw)],
            [(hhw,-10), (hhw,hhw), (-10,hhw)],
        ],
        obstacle_polygons=[],
        obstacle_paths=[],
        obstacle_x_low=0,
        obstacle_x_high=0,
        obstacle_y_low=0,
        obstacle_y_high=0,
        start_x=0,
        start_y=-7,
        start_V = 2.5,
        start_theta=float(np.radians(90)),
        start_x_noise = 0.2,
        start_y_noise = 0.2,
        start_V_noise = 2.5,
        start_theta_noise = float(np.radians(30)),
        goal_x=-7,
        goal_y=0,
        goal_theta=float(np.radians(180)),
        goal_region=Polyhedron(
            # x <= -7
            # y >= -hhw
            # y <= hhw
            A=np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 1, 0, 0],
            ]),
            b=np.array([-7, hhw, hhw]),
        ),
        center_reward_region=Polyhedron(
            # x <= -hhw
            # y >= -hhw
            # y <= hhw
            A=np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 1, 0, 0],
            ]),
            b=np.array([-hhw, hhw, hhw]),
        ),
        center_reward_path=((0, 0), (-10, 0)),
    )
    sharp_turn_corner_const = float(hhw * np.sqrt(3))
    sharp_turn_end_high = float((-10 + hall_width) / np.sqrt(3))
    sharp_turn_end_low = float((-10 - hall_width) / np.sqrt(3))
    sharp_right = F110ObstacleMode(
        name='f110_sharp_right',
        static_polygons=[],
        static_paths=[
            [(-hhw,-10), (-hhw,sharp_turn_corner_const), (10,sharp_turn_end_high)],
            [(hhw,-10), (hhw,-sharp_turn_corner_const), (10,sharp_turn_end_low)],
        ],
        obstacle_polygons=[],
        obstacle_paths=[],
        obstacle_x_low=0,
        obstacle_x_high=0,
        obstacle_y_low=0,
        obstacle_y_high=0,
        start_x=0,
        start_y=-7,
        start_V = 2.5,
        start_theta=float(np.radians(90)),
        start_x_noise = 0.2,
        start_y_noise = 0.2,
        start_V_noise = 2.5,
        start_theta_noise = float(np.radians(30)),
        goal_x=float(3.5 * np.sqrt(3)),
        goal_y=-3.5,
        goal_theta=float(np.radians(-30)),
        goal_region=Polyhedron(
            # (sqrt(3)/2) * x + (-1/2) * y >= 7
            # y >= -x/sqrt(3) - 2*hhw/sqrt(3)
            # y <= -x/sqrt(3) + 2*hhw/sqrt(3)
            A=np.array([
                [-0.5 * np.sqrt(3), 0.5, 0, 0],
                [-1/np.sqrt(3), -1, 0, 0],
                [1/np.sqrt(3), 1, 0, 0],
            ]),
            b=np.array([-7, 2*hhw/np.sqrt(3), 2*hhw/np.sqrt(3)]),
        ),
        center_reward_region=Polyhedron(
            # (sqrt(3)/2) * x + (-1/2) * y >= sqrt(3)*hhw
            # y >= -x/sqrt(3) - 2*hhw/sqrt(3)
            # y <= -x/sqrt(3) + 2*hhw/sqrt(3)
            A=np.array([
                [-0.5 * np.sqrt(3), 0.5, 0, 0],
                [-1/np.sqrt(3), -1, 0, 0],
                [1/np.sqrt(3), 1, 0, 0],
            ]),
            b=np.array([-np.sqrt(3)*hhw, 2*hhw/np.sqrt(3), 2*hhw/np.sqrt(3)]),
        ),
        center_reward_path=((0, 0), (10, float(-10/np.sqrt(3)))),
    )
    sharp_left = F110ObstacleMode(
        name='f110_sharp_left',
        static_polygons=[],
        static_paths=[
            [(-hhw,-10), (-hhw,-sharp_turn_corner_const), (-10,sharp_turn_end_low)],
            [(hhw,-10), (hhw,sharp_turn_corner_const), (-10,sharp_turn_end_high)],
        ],
        obstacle_polygons=[],
        obstacle_paths=[],
        obstacle_x_low=0,
        obstacle_x_high=0,
        obstacle_y_low=0,
        obstacle_y_high=0,
        start_x=0,
        start_y=-7,
        start_V = 2.5,
        start_theta=float(np.radians(90)),
        start_x_noise = 0.2,
        start_y_noise = 0.2,
        start_V_noise = 2.5,
        start_theta_noise = float(np.radians(30)),
        goal_x=float(-3.5 * np.sqrt(3)),
        goal_y=-3.5,
        goal_theta=float(np.radians(210)),
        goal_region=Polyhedron(
            # (-sqrt(3)/2) * x + (-1/2) * y >= 7
            # y >= x/sqrt(3) - 2*hhw/sqrt(3)
            # y <= x/sqrt(3) + 2*hhw/sqrt(3)
            A=np.array([
                [0.5 * np.sqrt(3), 0.5, 0, 0],
                [1/np.sqrt(3), -1, 0, 0],
                [-1/np.sqrt(3), 1, 0, 0],
            ]),
            b=np.array([-7, 2*hhw/np.sqrt(3), 2*hhw/np.sqrt(3)]),
        ),
        center_reward_region=Polyhedron(
            # (-sqrt(3)/2) * x + (-1/2) * y >= sqrt(3)*hhw
            # y >= x/sqrt(3) - 2*hhw/sqrt(3)
            # y <= x/sqrt(3) + 2*hhw/sqrt(3)
            A=np.array([
                [0.5 * np.sqrt(3), 0.5, 0, 0],
                [1/np.sqrt(3), -1, 0, 0],
                [-1/np.sqrt(3), 1, 0, 0],
            ]),
            b=np.array([-np.sqrt(3)*hhw, 2*hhw/np.sqrt(3), 2*hhw/np.sqrt(3)]),
        ),
        center_reward_path=((0, 0), (-10, float(-10/np.sqrt(3)))),
    )
    obstacle = F110ObstacleMode(
        name='f110_obstacle',
        static_polygons=[],
        static_paths=[
            [(-hhw,-12), (-hhw,-5), (-hall_width, -3), (-hall_width, 3), (-hhw, 5), (-hhw, 12)],
            [(hhw,-12), (hhw,-5), (hall_width, -3), (hall_width, 3), (hhw, 5), (hhw, 12)],
        ],
        obstacle_polygons=[[(0.5, 0.5), (0.5, -0.5), (-0.5, -0.5), (-0.5, 0.5)]],
        obstacle_paths=[],
        obstacle_x_low=-hhw,
        obstacle_x_high=hhw,
        obstacle_y_low=-2,
        obstacle_y_high=2,
        start_x=0,
        start_y=-11,
        start_V=2.5,
        start_theta=float(np.radians(90)),
        start_x_noise=0.2,
        start_y_noise=0.2,
        start_V_noise=2.5,
        start_theta_noise = float(np.radians(30)),
        goal_x=0,
        goal_y=7,
        goal_theta=float(np.radians(90)),
        goal_region=Polyhedron(
            # y >= 7
            # x >= -hhw
            # x <= hhw
            A=np.array([
                [0, -1, 0, 0],
                [-1, 0, 0, 0],
                [1, 0, 0, 0],
            ]),
            b=np.array([-7, hhw, hhw]),
        ),
        center_reward_region=Polyhedron(
            # y >= 5
            # x >= -hhw
            # x <= hhw
            A=np.array([
                [0, -1, 0, 0],
                [-1, 0, 0, 0],
                [1, 0, 0, 0],
            ]),
            b=np.array([-5, hhw, hhw]),
        ),
        center_reward_path=((0, 0), (0, 20)),
    )
    modes = straight_modes + [square_right, square_left, sharp_right, sharp_left, obstacle]
    mode_dict = {mode.name: mode for mode in modes}
    return HybridAutomaton(
        modes=modes,
        transitions=[
            F110ObstacleTrans(source=m.name, modes=mode_dict)
            for m in modes
        ],
    )
