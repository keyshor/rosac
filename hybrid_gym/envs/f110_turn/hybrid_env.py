from hybrid_gym.hybrid_env import HybridAutomaton
from hybrid_gym.envs.f110.transition import F110ObstacleTrans
from hybrid_gym.envs.f110_turn.mode import (
    make_straight,
    make_square_right, make_square_left,
    make_sharp_right, make_sharp_left,
)
from typing import Iterable

def make_f110_model(straight_lengths: Iterable[float] = [10.0],
                    use_throttle: bool = False,
                    num_lidar_rays: int = 1081,
                    hall_width: float = 1.5,
                    goal_guide: bool = False,
                    ) -> HybridAutomaton:
    modes = [
        make_straight(
            length=l,
            use_throttle=use_throttle,
            num_lidar_rays=num_lidar_rays,
            width=hall_width,
            goal_guide=goal_guide,
        )
        for l in straight_lengths
    ] + [
        make_square_right(
            use_throttle=use_throttle,
            num_lidar_rays=num_lidar_rays,
            width=hall_width,
            goal_guide=goal_guide,
        ),
        make_square_left(
            use_throttle=use_throttle,
            num_lidar_rays=num_lidar_rays,
            width=hall_width,
            goal_guide=goal_guide,
        ),
        make_sharp_right(
            use_throttle=use_throttle,
            num_lidar_rays=num_lidar_rays,
            width=hall_width,
            goal_guide=goal_guide,
        ),
        make_sharp_left(
            use_throttle=use_throttle,
            num_lidar_rays=num_lidar_rays,
            width=hall_width,
            goal_guide=goal_guide,
        ),
    ]
    return HybridAutomaton(
        modes=modes,
        transitions=[
            F110ObstacleTrans(
                source=src.name,
                plain_modes={},
                obstacle_modes={
                    m.name: m
                    for m in modes
                },
            )
            for src in modes
        ],
    )
