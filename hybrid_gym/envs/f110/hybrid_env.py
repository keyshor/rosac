from hybrid_gym.hybrid_env import HybridAutomaton
from hybrid_gym.envs.f110.mode import (
    make_straight,
    make_square_right, make_square_left,
    make_sharp_right, make_sharp_left
)
from hybrid_gym.envs.f110.transition import F110Trans
from typing import Iterable


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

    modes = [
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

    if not simple:
        modes += [
            make_sharp_right(use_throttle=use_throttle,
                             lidar_num_rays=num_lidar_rays,
                             width=hall_width),
            make_sharp_left(use_throttle=use_throttle,
                            lidar_num_rays=num_lidar_rays,
                            width=hall_width),
        ]

    f110_automaton = HybridAutomaton(modes=modes,
                                     transitions=[
                                         F110Trans(source=src.name, modes={
                                                   m.name: m for m in modes}, hall_width=hall_width)
                                         for src in modes])

    return f110_automaton
