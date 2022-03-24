import numpy as np
from hybrid_gym.envs.f110.obstacle_mode import F110ObstacleMode, Polyhedron, LIDAR_RANGE

def make_straight(use_throttle: bool,
                  num_lidar_rays: int,
                  width: float,
                  length: float,
                  goal_guide: bool,
                  ) -> F110ObstacleMode:
    hhw = width / 2
    return F110ObstacleMode(
        name='f110_straight',
        static_polygons=[],
        static_paths=[
            [(-hhw, -length - LIDAR_RANGE), (-hhw, LIDAR_RANGE)],
            [(hhw, -length - LIDAR_RANGE), (hhw, LIDAR_RANGE)],
        ],
        obstacle_polygons=[],
        obstacle_paths=[],
        obstacle_x_low=-1.0,
        obstacle_x_high=1.0,
        obstacle_y_low=-1.0,
        obstacle_y_high=1.0,
        start_x=0,
        start_y=-length,
        start_V=2.5,
        start_theta=float(np.radians(90)),
        start_x_noise=0.2,
        start_y_noise=0.2,
        start_V_noise=0.5,
        start_theta_noise = float(np.radians(30)),
        goal_x=0,
        goal_y=0,
        goal_theta=float(np.radians(90)),
        use_throttle=use_throttle,
        num_lidar_rays=num_lidar_rays,
        goal_region=Polyhedron(
            # y >= 0
            # x >= -hhw
            # x <= hhw
            A=np.array([
                [0, -1, 0, 0],
                [-1, 0, 0, 0],
                [1, 0, 0, 0],
            ]),
            b=np.array([0, hhw, hhw]),
        ),
        center_reward_region=Polyhedron(
            # y >= -length - LIDAR_RANGE
            # x >= -hhw
            # x <= hhw
            A=np.array([
                [0, -1, 0, 0],
                [-1, 0, 0, 0],
                [1, 0, 0, 0],
            ]),
            b=np.array([length + LIDAR_RANGE, hhw, hhw]),
        ) if goal_guide else Polyhedron(
            # 0 <= -1
            A=np.zeros(shape=(1,4)),
            b=np.array([-1]),
        ),
        center_reward_path=((0, -length - LIDAR_RANGE), (0, LIDAR_RANGE)),
    )

def make_square_right(use_throttle: bool,
                      num_lidar_rays: int,
                      width: float,
                      goal_guide: bool,
                      ) -> F110ObstacleMode:
    hhw = width / 2
    return F110ObstacleMode(
        name='f110_square_right',
        static_polygons=[],
        static_paths=[
            [(-hhw, -2*LIDAR_RANGE), (-hhw, hhw), (LIDAR_RANGE + 2, hhw)],
            [(hhw, -2*LIDAR_RANGE), (hhw, -hhw), (LIDAR_RANGE + 2, -hhw)],
        ],
        obstacle_polygons=[],
        obstacle_paths=[],
        obstacle_x_low=-1.0,
        obstacle_x_high=1.0,
        obstacle_y_low=-1.0,
        obstacle_y_high=1.0,
        start_x=0,
        start_y=-LIDAR_RANGE - 1.0,
        start_V=2.5,
        start_theta=float(np.radians(90)),
        start_x_noise=0.2,
        start_y_noise=0.2,
        start_V_noise=0.5,
        start_theta_noise = float(np.radians(30)),
        goal_x=hhw,
        goal_y=0,
        goal_theta=0.0,
        use_throttle=use_throttle,
        num_lidar_rays=num_lidar_rays,
        goal_region=Polyhedron(
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
        center_reward_region=Polyhedron(
            # x >= 0
            # y >= -hhw
            # y <= hhw
            A=np.array([
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 1, 0, 0],
            ]),
            b=np.array([0, hhw, hhw]),
        ) if goal_guide else Polyhedron(
            # 0 <= -1
            A=np.zeros(shape=(1,4)),
            b=np.array([-1]),
        ),
        center_reward_path=((0, 0), (LIDAR_RANGE, 0)),
    )

def make_square_left(use_throttle: bool,
                     num_lidar_rays: int,
                     width: float,
                     goal_guide: bool,
                     ) -> F110ObstacleMode:
    hhw = width / 2
    return F110ObstacleMode(
        name='f110_square_left',
        static_polygons=[],
        static_paths=[
            [(-hhw, -2*LIDAR_RANGE), (-hhw, -hhw), (-LIDAR_RANGE - 2, -hhw)],
            [(hhw, -2*LIDAR_RANGE), (hhw, hhw), (-LIDAR_RANGE - 2, hhw)],
        ],
        obstacle_polygons=[],
        obstacle_paths=[],
        obstacle_x_low=-1.0,
        obstacle_x_high=1.0,
        obstacle_y_low=-1.0,
        obstacle_y_high=1.0,
        start_x=0,
        start_y=-LIDAR_RANGE - 1.0,
        start_V=2.5,
        start_theta=float(np.radians(90)),
        start_x_noise=0.2,
        start_y_noise=0.2,
        start_V_noise=0.5,
        start_theta_noise = float(np.radians(30)),
        goal_x=-hhw,
        goal_y=0,
        goal_theta=float(np.radians(180)),
        use_throttle=use_throttle,
        num_lidar_rays=num_lidar_rays,
        goal_region=Polyhedron(
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
        center_reward_region=Polyhedron(
            # x <= 0
            # y >= -hhw
            # y <= hhw
            A=np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 1, 0, 0],
            ]),
            b=np.array([0, hhw, hhw]),
        ) if goal_guide else Polyhedron(
            # 0 <= -1
            A=np.zeros(shape=(1,4)),
            b=np.array([-1]),
        ),
        center_reward_path=((0, 0), (-LIDAR_RANGE, 0)),
    )

def make_sharp_right(use_throttle: bool,
                     num_lidar_rays: int,
                     width: float,
                     goal_guide: bool,
                     ) -> F110ObstacleMode:
    hhw = width / 2
    sqrt3 = np.sqrt(3)
    w_div_sqrt3 = width / sqrt3
    return F110ObstacleMode(
        name='f110_sharp_right',
        static_polygons=[],
        static_paths=[
            [
                (-hhw, -2*LIDAR_RANGE),
                (-hhw, hhw),
                ((2 - sqrt3) * hhw, hhw),
                (LIDAR_RANGE, (-LIDAR_RANGE + width) / sqrt3),
            ],
            [
                (hhw, -2*LIDAR_RANGE),
                (hhw, -sqrt3 * hhw),
                (LIDAR_RANGE, (-LIDAR_RANGE - width) / sqrt3),
            ],
        ],
        obstacle_polygons=[],
        obstacle_paths=[],
        obstacle_x_low=-1.0,
        obstacle_x_high=1.0,
        obstacle_y_low=-1.0,
        obstacle_y_high=1.0,
        start_x=0,
        start_y=-LIDAR_RANGE - 1.0,
        start_V=2.5,
        start_theta=float(np.radians(90)),
        start_x_noise=0.2,
        start_y_noise=0.2,
        start_V_noise=0.5,
        start_theta_noise = float(np.radians(30)),
        goal_x=hhw,
        goal_y=-hhw / sqrt3,
        goal_theta=float(np.radians(-30)),
        use_throttle=use_throttle,
        num_lidar_rays=num_lidar_rays,
        goal_region=Polyhedron(
            # x >= hhw
            # y >= (-x - w) / sqrt(3)
            # y <= (-x + w) / sqrt(3)
            A=np.array([
                [-1, 0, 0, 0],
                [-1/sqrt3, -1, 0, 0],
                [1/sqrt3, 1, 0, 0],
            ]),
            b=np.array([-hhw, w_div_sqrt3, w_div_sqrt3]),
        ),
        center_reward_region=Polyhedron(
            # x >= 0
            # y >= (-x - w) / sqrt(3)
            # y <= (-x + w) / sqrt(3)
            A=np.array([
                [-1, 0, 0, 0],
                [-1/sqrt3, -1, 0, 0],
                [1/sqrt3, 1, 0, 0],
            ]),
            b=np.array([0, w_div_sqrt3, w_div_sqrt3]),
        ) if goal_guide else Polyhedron(
            # 0 <= -1
            A=np.zeros(shape=(1,4)),
            b=np.array([-1]),
        ),
        center_reward_path=((0, 0), (LIDAR_RANGE, -LIDAR_RANGE / sqrt3)),
    )

def make_sharp_left(use_throttle: bool,
                    num_lidar_rays: int,
                    width: float,
                    goal_guide: bool,
                    ) -> F110ObstacleMode:
    hhw = width / 2
    sqrt3 = np.sqrt(3)
    w_div_sqrt3 = width / sqrt3
    return F110ObstacleMode(
        name='f110_sharp_left',
        static_polygons=[],
        static_paths=[
            [
                (-hhw, -2*LIDAR_RANGE),
                (-hhw, -hhw * sqrt3),
                (-LIDAR_RANGE, (-LIDAR_RANGE - width) / sqrt3),
            ],
            [
                (hhw, -2*LIDAR_RANGE),
                (hhw, hhw),
                ((sqrt3 - 2) * hhw, hhw),
                (-LIDAR_RANGE, (-LIDAR_RANGE + width) / sqrt3),
            ],
        ],
        obstacle_polygons=[],
        obstacle_paths=[],
        obstacle_x_low=-1.0,
        obstacle_x_high=1.0,
        obstacle_y_low=-1.0,
        obstacle_y_high=1.0,
        start_x=0,
        start_y=-LIDAR_RANGE - 1.0,
        start_V=2.5,
        start_theta=float(np.radians(90)),
        start_x_noise=0.2,
        start_y_noise=0.2,
        start_V_noise=0.5,
        start_theta_noise = float(np.radians(30)),
        goal_x=-hhw,
        goal_y=-hhw / sqrt3,
        goal_theta=float(np.radians(210)),
        use_throttle=use_throttle,
        num_lidar_rays=num_lidar_rays,
        goal_region=Polyhedron(
            # x <= -hhw
            # y >= (x - w) / sqrt(3)
            # y <= (x + w) / sqrt(3)
            A=np.array([
                [1, 0, 0, 0],
                [1/sqrt3, -1, 0, 0],
                [-1/sqrt3, 1, 0, 0],
            ]),
            b=np.array([-hhw, w_div_sqrt3, w_div_sqrt3]),
        ),
        center_reward_region=Polyhedron(
            # x <= 0
            # y >= (x - w) / sqrt(3)
            # y <= (x + w) / sqrt(3)
            A=np.array([
                [1, 0, 0, 0],
                [1/sqrt3, -1, 0, 0],
                [-1/sqrt3, 1, 0, 0],
            ]),
            b=np.array([0, w_div_sqrt3, w_div_sqrt3]),
        ) if goal_guide else Polyhedron(
            # 0 <= -1
            A=np.zeros(shape=(1,4)),
            b=np.array([-1]),
        ),
        center_reward_path=((0, 0), (-LIDAR_RANGE, -LIDAR_RANGE / sqrt3)),
    )
