import numpy as np
from typing import Dict, Tuple

from hybrid_gym.model import Transition
from hybrid_gym.envs.rooms_two_doors.mode import RoomsMode


class RoomsTrans(Transition):
    modes: Dict[str, RoomsMode]

    def __init__(self,
                 source: str,
                 modes: Dict[str, RoomsMode],
                 ) -> None:
        self.modes = modes
        self.source_mode = self.modes[source]
        super().__init__(source, list(modes.keys()))

    def guard(self, state: Tuple[Tuple, Tuple]) -> bool:
        return (self.source_mode.completed_task(np.array(state[1])) == self.source) and \
            self.source_mode.is_safe(state)

    def jump(self, target: str, state: Tuple[Tuple, Tuple]) -> Tuple[Tuple, Tuple]:
        s1 = self.source_mode.mode_transition(np.array(state[0]), self.source)
        s2 = self.source_mode.mode_transition(np.array(state[1]), self.source)
        return (tuple(s1), tuple(s2))
