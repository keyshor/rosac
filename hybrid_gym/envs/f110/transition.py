from hybrid_gym.model import Transition
from hybrid_gym.envs.f110.mode import LIDAR_RANGE, State, F110Mode
from typing import Dict


def straight_length(mode_name: str) -> float:
    length_str = mode_name[len('f110_straight_'):-1]
    if mode_name != 'f110_straight_' + length_str + 'm':
        return 0
    try:
        return float(length_str)
    except ValueError:
        return 0


def flip_sides(mode1: str, mode2: str):
    return ('left' in mode1) != ('left' in mode2)


class F110Trans(Transition):
    modes: Dict[str, F110Mode]
    src_is_turn: bool
    hall_width: float

    def __init__(self,
                 source: str,
                 modes: Dict[str, F110Mode],
                 hall_width: float,
                 ) -> None:
        self.modes = modes
        self.src_is_turn = ('straight' not in source)
        self.hall_width = hall_width
        super().__init__(source, [m.name for m in modes.values()])

    def guard(self, st: State) -> bool:
        if 'straight' in self.source:
            return st.curHall == 0 and \
                st.car_dist_f <= LIDAR_RANGE + 2
        else:
            return st.curHall == 1

    def jump(self, target: str, st: State) -> State:
        dist = straight_length(target)
        new_dist_f = st.car_dist_f - (7 if self.src_is_turn else 0)
        if flip_sides(self.source, target):
            car_dist_s = self.hall_width - st.car_dist_s
        else:
            car_dist_s = st.car_dist_s
        return self.modes[target].set_state_local(
            car_dist_s, dist + new_dist_f, st.car_heading, st)
