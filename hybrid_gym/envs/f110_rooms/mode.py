import numpy as np
from typing import Optional, Tuple
from hybrid_gym.envs.f110.obstacle_mode import F110ObstacleMode, Polyhedron


def make_room_left(use_throttle: bool,
                   lidar_num_rays: int,
                   hall_width: float,
                   hall_length: float,
                   room_width: float,
                   observe_heading: bool,
                   mode_onehot_indices: Optional[Tuple[int, int]],
                   bad_start_region: bool,
                   ) -> F110ObstacleMode:
    assert hall_width <= room_width
    hhw = hall_width / 2
    r = room_width / 2
    override_start_region = (-0.8*r, 0.8*r, -0.8*r, -0.7*r, float(np.radians(-60)), float(np.radians(240)))
    return F110ObstacleMode(
        name='f110_room_left',
        static_polygons=[],
        static_paths=[
            [(-hhw, -r - hall_length), (-hhw, -r), (-r, -r), (-r, -hhw), (-r - hall_length, -hhw)],
            [(-r - hall_length, hhw), (-r, hhw), (-r, r), (-hhw, r), (-hhw, r + hall_length)],
            [(hhw, r + hall_length), (hhw, r), (r, r), (r, hhw), (r + hall_length, hhw)],
            [(r + hall_length, -hhw), (r, -hhw), (r, -r), (hhw, -r), (hhw, -r - hall_length)],
        ],
        obstacle_polygons=[[(0.5, 0.1), (0.5, -0.1), (-0.5, -0.1), (-0.5, 0.1)]],
        #obstacle_polygons=[],
        obstacle_paths=[],
        obstacle_x_low=-hhw,
        obstacle_x_high=hhw,
        obstacle_y_low=-0.6*r,
        obstacle_y_high=-0.5*r,
        start_x=0,
        start_y=-r - 0.5 * hall_length,
        start_V=2.5,
        start_theta=float(np.radians(90)),
        start_x_noise=0.2 * hhw,
        start_y_noise=0.2,
        start_V_noise=2.5,
        start_theta_noise = float(np.radians(30)),
        goal_x=-r - 0.6 * hall_length,
        goal_y=0,
        goal_theta=float(np.radians(180)),
        goal_region=Polyhedron(
            # x <= -r - 0.5 * hall_length
            # y >= -hhw
            # y <= hhw
            A=np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 1, 0, 0],
            ]),
            b=np.array([-r - 0.5 * hall_length, hhw, hhw]),
        ),
        center_reward_region=Polyhedron(
            # x <= -r
            # y >= -hhw
            # y <= hhw
            A=np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 1, 0, 0],
            ]),
            b=np.array([-r, hhw, hhw]),
        ),
        center_reward_path=((0, 0), (-r - hall_length, 0)),
        override_start_region=override_start_region if bad_start_region else None,
        additional_unsafe_regions=[
            Polyhedron(
                # y >= r + 0.7 * hall_length
                # x >= -hhw
                # x <= hhw
                A=np.array([
                    [0, -1, 0, 0],
                    [-1, 0, 0, 0],
                    [1, 0, 0, 0],
                ]),
                b=np.array([-r - 0.7 * hall_length, hhw, hhw]),
            ),
            Polyhedron(
                # x >= r + 0.7 * hall_length
                # y >= -hhw
                # y <= hhw
                A=np.array([
                    [-1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 1, 0, 0],
                ]),
                b=np.array([-r - 0.7 * hall_length, hhw, hhw]),
            ),
            Polyhedron(
                # y <= -r - 0.7 * hall_length
                # x >= -hhw
                # x <= hhw
                A=np.array([
                    [0, 1, 0, 0],
                    [-1, 0, 0, 0],
                    [1, 0, 0, 0],
                ]),
                b=np.array([-r - 0.7 * hall_length, hhw, hhw]),
            ),
        ],
        observe_heading=observe_heading,
        mode_onehot_indices=mode_onehot_indices,
    )

def make_room_right(use_throttle: bool,
                    lidar_num_rays: int,
                    hall_width: float,
                    hall_length: float,
                    room_width: float,
                    observe_heading: bool,
                    mode_onehot_indices: Optional[Tuple[int, int]],
                    bad_start_region: bool,
                    ) -> F110ObstacleMode:
    assert hall_width <= room_width
    hhw = hall_width / 2
    r = room_width / 2
    override_start_region = (-0.8*r, 0.8*r, -0.8*r, -0.7*r, float(np.radians(-60)), float(np.radians(240)))
    return F110ObstacleMode(
        name='f110_room_right',
        static_polygons=[],
        static_paths=[
            [(-hhw, -r - hall_length), (-hhw, -r), (-r, -r), (-r, -hhw), (-r - hall_length, -hhw)],
            [(-r - hall_length, hhw), (-r, hhw), (-r, r), (-hhw, r), (-hhw, r + hall_length)],
            [(hhw, r + hall_length), (hhw, r), (r, r), (r, hhw), (r + hall_length, hhw)],
            [(r + hall_length, -hhw), (r, -hhw), (r, -r), (hhw, -r), (hhw, -r - hall_length)],
        ],
        obstacle_polygons=[[(0.5, 0.1), (0.5, -0.1), (-0.5, -0.1), (-0.5, 0.1)]],
        #obstacle_polygons=[],
        obstacle_paths=[],
        obstacle_x_low=-hhw,
        obstacle_x_high=hhw,
        obstacle_y_low=-r + 1.0,
        obstacle_y_high=-r + 1.5,
        start_x=0,
        start_y=-r - 0.5 * hall_length,
        start_V=2.5,
        start_theta=float(np.radians(90)),
        start_x_noise=0.2 * hhw,
        start_y_noise=0.2,
        start_V_noise=2.5,
        start_theta_noise = float(np.radians(30)),
        goal_x=r + 0.6 * hall_length,
        goal_y=0,
        goal_theta=0,
        goal_region=Polyhedron(
            # x >= r + 0.5 * hall_length
            # y >= -hhw
            # y <= hhw
            A=np.array([
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 1, 0, 0],
            ]),
            b=np.array([-r - 0.5 * hall_length, hhw, hhw]),
        ),
        center_reward_region=Polyhedron(
            # x >= r
            # y >= -hhw
            # y <= hhw
            A=np.array([
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 1, 0, 0],
            ]),
            b=np.array([-r, hhw, hhw]),
        ),
        center_reward_path=((0, 0), (r + hall_length, 0)),
        override_start_region=override_start_region if bad_start_region else None,
        additional_unsafe_regions=[
            Polyhedron(
                # y >= r + 0.7 * hall_length
                # x >= -hhw
                # x <= hhw
                A=np.array([
                    [0, -1, 0, 0],
                    [-1, 0, 0, 0],
                    [1, 0, 0, 0],
                ]),
                b=np.array([-r - 0.7 * hall_length, hhw, hhw]),
            ),
            Polyhedron(
                # x <= -r - 0.7 * hall_length
                # y >= -hhw
                # y <= hhw
                A=np.array([
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 1, 0, 0],
                ]),
                b=np.array([-r - 0.7 * hall_length, hhw, hhw]),
            ),
            Polyhedron(
                # y <= -r - 0.7 * hall_length
                # x >= -hhw
                # x <= hhw
                A=np.array([
                    [0, 1, 0, 0],
                    [-1, 0, 0, 0],
                    [1, 0, 0, 0],
                ]),
                b=np.array([-r - 0.7 * hall_length, hhw, hhw]),
            ),
        ],
        observe_heading=observe_heading,
        mode_onehot_indices=mode_onehot_indices,
    )

def make_room_up(use_throttle: bool,
                 lidar_num_rays: int,
                 hall_width: float,
                 hall_length: float,
                 room_width: float,
                 observe_heading: bool,
                 mode_onehot_indices: Optional[Tuple[int, int]],
                 bad_start_region: bool,
                 ) -> F110ObstacleMode:
    assert hall_width <= room_width
    hhw = hall_width / 2
    r = room_width / 2
    override_start_region = (-0.8*r, 0.8*r, -0.8*r, -0.7*r, float(np.radians(-60)), float(np.radians(240)))
    return F110ObstacleMode(
        name='f110_room_up',
        static_polygons=[],
        static_paths=[
            [(-hhw, -r - hall_length), (-hhw, -r), (-r, -r), (-r, -hhw), (-r - hall_length, -hhw)],
            [(-r - hall_length, hhw), (-r, hhw), (-r, r), (-hhw, r), (-hhw, r + hall_length)],
            [(hhw, r + hall_length), (hhw, r), (r, r), (r, hhw), (r + hall_length, hhw)],
            [(r + hall_length, -hhw), (r, -hhw), (r, -r), (hhw, -r), (hhw, -r - hall_length)],
        ],
        obstacle_polygons=[[(0.5, 0.1), (0.5, -0.1), (-0.5, -0.1), (-0.5, 0.1)]],
        #obstacle_polygons=[],
        obstacle_paths=[],
        obstacle_x_low=-hhw,
        obstacle_x_high=hhw,
        obstacle_y_low=-r + 1.0,
        obstacle_y_high=-r + 1.5,
        start_x=0,
        start_y=-r - 0.5 * hall_length,
        start_V=2.5,
        start_theta=float(np.radians(90)),
        start_x_noise=0.2 * hhw,
        start_y_noise=0.2,
        start_V_noise=2.5,
        start_theta_noise = float(np.radians(30)),
        goal_x=0,
        goal_y=r + 0.6 * hall_length,
        goal_theta=float(np.radians(90)),
        goal_region=Polyhedron(
            # y >= r + 0.5 * hall_length
            # x >= -hhw
            # x <= hhw
            A=np.array([
                [0, -1, 0, 0],
                [-1, 0, 0, 0],
                [1, 0, 0, 0],
            ]),
            b=np.array([-r - 0.5 * hall_length, hhw, hhw]),
        ),
        center_reward_region=Polyhedron(
            # y >= r
            # x >= -hhw
            # x <= hhw
            A=np.array([
                [0, -1, 0, 0],
                [-1, 0, 0, 0],
                [1, 0, 0, 0],
            ]),
            b=np.array([-r, hhw, hhw]),
        ),
        center_reward_path=((0, 0), (0, r + hall_length)),
        override_start_region=override_start_region if bad_start_region else None,
        additional_unsafe_regions=[
            Polyhedron(
                # x <= -r - 0.7 * hall_length
                # y >= -hhw
                # y <= hhw
                A=np.array([
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 1, 0, 0],
                ]),
                b=np.array([-r - 0.7 * hall_length, hhw, hhw]),
            ),
            Polyhedron(
                # x >= r + 0.7 * hall_length
                # y >= -hhw
                # y <= hhw
                A=np.array([
                    [-1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 1, 0, 0],
                ]),
                b=np.array([-r - 0.7 * hall_length, hhw, hhw]),
            ),
            Polyhedron(
                # y <= -r - 0.7 * hall_length
                # x >= -hhw
                # x <= hhw
                A=np.array([
                    [0, 1, 0, 0],
                    [-1, 0, 0, 0],
                    [1, 0, 0, 0],
                ]),
                b=np.array([-r - 0.7 * hall_length, hhw, hhw]),
            ),
        ],
        observe_heading=observe_heading,
        mode_onehot_indices=mode_onehot_indices,
    )
