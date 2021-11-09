import numpy as np
from hybrid_gym.hybrid_env import HybridAutomaton
from hybrid_gym.envs.f110.obstacle_mode import F110ObstacleMode
from hybrid_gym.envs.f110_rooms.mode import make_room_left, make_room_right, make_room_up
from hybrid_gym.envs.f110_rooms.transition import F110ObstacleTrans

def make_f110_rooms_model(use_throttle: bool = True,
                          observe_heading: bool = True,
                          lidar_num_rays: int = 1081,
                          hall_width: float = 1.5,
                          hall_length: float = 5,
                          room_width: float = 5,
                          observe_mode_onehot: bool = False,
                          bad_start_region: bool = False,
                          ) -> HybridAutomaton:
    modes = [
        make_room_left(
            use_throttle=use_throttle, lidar_num_rays=lidar_num_rays,
            hall_width=hall_width, hall_length=hall_length, room_width=room_width,
            observe_heading=observe_heading, bad_start_region=bad_start_region,
            mode_onehot_indices=(3, 0) if observe_mode_onehot else None,
        ),
        make_room_right(
            use_throttle=use_throttle, lidar_num_rays=lidar_num_rays,
            hall_width=hall_width, hall_length=hall_length, room_width=room_width,
            observe_heading=observe_heading, bad_start_region=bad_start_region,
            mode_onehot_indices=(3, 1) if observe_mode_onehot else None,
        ),
        make_room_up(
            use_throttle=use_throttle, lidar_num_rays=lidar_num_rays,
            hall_width=hall_width, hall_length=hall_length, room_width=room_width,
            observe_heading=observe_heading, bad_start_region=bad_start_region,
            mode_onehot_indices=(3, 2) if observe_mode_onehot else None,
        ),
    ]
    mode_dict = {mode.name: mode for mode in modes}
    return HybridAutomaton(
        modes=modes,
        transitions=[
            F110ObstacleTrans(source=m.name, modes=mode_dict)
            for m in modes
        ],
    )
