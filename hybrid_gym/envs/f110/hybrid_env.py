from hybrid_gym.hybrid_env import HybridAutomaton
from hybrid_gym.envs.f110.mode import (
    make_straight,
    make_square_right, make_square_left,
    make_sharp_right, make_sharp_left
)
from hybrid_gym.envs.f110.transition import F110Trans
from typing import Iterable


def make_f110_model(straight_lengths: Iterable[float] = [10.0],
                    num_lidar_rays: int = 1081
                    ) -> HybridAutomaton:
    modes = [
        make_straight(lnth, num_lidar_rays) for lnth in straight_lengths
    ] + [
        make_square_right(num_lidar_rays),
        make_square_left(num_lidar_rays),
        make_sharp_right(num_lidar_rays),
        make_sharp_left(num_lidar_rays),
    ]

    f110_automaton = HybridAutomaton(
        modes=modes,
        transitions=[
            F110Trans(source=src.name, modes={m.name: m for m in modes})
            for src in modes
        ]
    )

    return f110_automaton
