from hybrid_gym.model import Transition
from hybrid_gym.envs.f110.mode import (
    LIDAR_RANGE, State, F110Mode
)
from typing import Dict, Optional


def straight_length(mode_name: str) -> Optional[float]:
    length_str = mode_name[len('f110_straight_'):-1]
    if mode_name != 'f110_straight_' + length_str + 'm':
        return None
    try:
        return float(length_str)
    except ValueError:
        return None


class F110Trans(Transition):
    modes: Dict[str, F110Mode]

    def __init__(self, source: str, modes: Dict[str, F110Mode]) -> None:
        self.modes = modes
        super().__init__(source, [m.name for m in modes.values()])

    def guard(self, st: State) -> bool:
        if 'straight' in self.source:
            return st.curHall == 0 and \
                st.car_dist_f <= LIDAR_RANGE + 2
        else:
            return st.curHall == 1

    def jump(self, target: str, st: State) -> State:
        dist = straight_length(target)
        if dist is None:
            return self.modes[target].set_state_local(
                st.car_dist_s, st.car_dist_f, st.car_heading, st
            )
        else:
            return self.modes[target].set_state_local(
                st.car_dist_s,
                dist + st.car_dist_f,
                st.car_heading,
                st
            )
