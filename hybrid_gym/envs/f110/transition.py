from hybrid_gym.model import Transition
from hybrid_gym.envs.f110.mode import (
    LIDAR_RANGE, State, F110Mode, make_straight, normal_right
)
from typing import Iterable, Dict, Tuple, Final

straight_10 = make_straight(10)
straight_15 = make_straight(15)

straight_modes: Final[Dict[str, Tuple[float, F110Mode]]] = {
    straight_10.name: (10, straight_10),
    straight_15.name: (15, straight_15),
}


class F110Trans(Transition):
    def guard(self, st: State) -> bool:
        if self.source in straight_modes:
            return st.curHall == 0 and \
                st.car_dist_f <= LIDAR_RANGE + 2
        else:
            return st.curHall == 1

    def jump(self, target: str, st: State) -> State:
        try:
            dist, mode = straight_modes[target]
            return mode.set_state_local(
                st.car_dist_s,
                LIDAR_RANGE + dist + 2,
                st.car_heading,
                st
            )
        except KeyError:
            return normal_right.set_state_local(
                st.car_dist_s, LIDAR_RANGE + 2, st.car_heading, st
            )
