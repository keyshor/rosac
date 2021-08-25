from hybrid_gym.hybrid_env import HybridAutomaton
from hybrid_gym.envs.f110.mode import (
    F110Mode,
    make_straight,
    make_square_right, make_square_left,
    make_sharp_right, make_sharp_left
)
from hybrid_gym.envs.f110.obstacle_mode import F110ObstacleMode, make_obstacle
from hybrid_gym.envs.f110.transition import F110PlainTrans, F110ObstacleTrans
from typing import List, Iterable, Union


def cast_to_mode_union(l: List[F110Mode]) -> List[Union[F110Mode, F110ObstacleMode]]:
    return [x for x in l]

def cast_to_trans_union(l: List[F110PlainTrans]) -> List[Union[F110PlainTrans, F110ObstacleTrans]]:
    return [x for x in l]

def make_f110_model(straight_lengths: Iterable[float] = [10.0],
                    use_throttle: bool = True,
                    num_lidar_rays: int = 1081,
                    hall_width: float = 1.5,
                    simple: bool = False,
                    ) -> HybridAutomaton:
    '''
    Makes race track environment with f1/10th car model.

    straight lengths: list of possible lengths of straight segments.
    num_lidar_rays: number of lidar rays.
    hall_width: width of each hallway.
    '''

    plain_modes = [
        make_straight(length=lnth,
                      use_throttle=use_throttle,
                      lidar_num_rays=num_lidar_rays,
                      width=hall_width)
        for lnth in straight_lengths
    ] + [
        make_square_right(use_throttle=use_throttle,
                          lidar_num_rays=num_lidar_rays,
                          width=hall_width),
        make_square_left(use_throttle=use_throttle,
                         lidar_num_rays=num_lidar_rays,
                         width=hall_width),
    ]

    obstacle_mode = make_obstacle(use_throttle=use_throttle,
                                  lidar_num_rays=num_lidar_rays,
                                  width=hall_width)

    if simple:
        modes = cast_to_mode_union(plain_modes)
        transitions = cast_to_trans_union([
            F110PlainTrans(source=src.name,
                           plain_modes={m.name: m for m in plain_modes},
                           obstacle_modes={},
                           hall_width=hall_width)
            for src in plain_modes
        ])
    else:
        plain_modes += [
            make_sharp_right(use_throttle=use_throttle,
                             lidar_num_rays=num_lidar_rays,
                             width=hall_width),
            make_sharp_left(use_throttle=use_throttle,
                            lidar_num_rays=num_lidar_rays,
                            width=hall_width),
        ]
        modes = cast_to_mode_union(plain_modes) + [obstacle_mode]
        transitions = cast_to_trans_union([
            F110PlainTrans(source=src.name,
                           plain_modes={m.name: m for m in plain_modes},
                           obstacle_modes={obstacle_mode.name: obstacle_mode},
                           hall_width=hall_width)
            for src in plain_modes
        ]) + [
            F110ObstacleTrans(source=obstacle_mode.name,
                              plain_modes={m.name: m for m in plain_modes},
                              obstacle_modes={obstacle_mode.name: obstacle_mode})
        ]

    f110_automaton = HybridAutomaton(modes=modes, transitions=transitions)
    return f110_automaton
