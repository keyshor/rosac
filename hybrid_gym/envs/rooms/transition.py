import numpy as np
from typing import Dict

from hybrid_gym.model import Transition
from hybrid_gym.envs.rooms.mode import RoomsMode, State, Direction

class RoomsTrans(Transition):
    modes: Dict[str, RoomsMode]

    def __init__(self,
                 source: str,
                 modes: Dict[str, RoomsMode],
                 ) -> None:
        self.modes = modes
        super().__init__(source, list(modes.keys()))

    def guard(self, st: State) -> bool:
        goal_x_low, goal_x_high, goal_y_low, goal_y_high = self.modes[self.source].goal_region
        return goal_x_low <= st.x <= goal_x_high and goal_y_low <= st.y <= goal_y_high

    def jump(self, target: str, st: State) -> State:
        d = self.modes[self.source].direction
        adjustment = self.modes[self.source].room_radius + self.modes[target].room_radius
        return State(x = st.x, y = st.y - adjustment) if d == Direction.UP \
            else State(x = st.x - adjustment, y = st.y) if d == Direction.RIGHT \
            else State(x = st.x, y = st.y + adjustment) if d == Direction.DOWN \
            else State(x = st.x + adjustment, y = st.y)
