import numpy as np
from typing import Dict, Tuple

from hybrid_gym.model import Transition
from hybrid_gym.envs.rooms.mode import RoomsMode


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
        return self.source_mode.completed_task(np.array(state[1])) == self.source

    def jump(self, target: str, state: Tuple[Tuple, Tuple]) -> Tuple[Tuple, Tuple]:
        s = self.source_mode.mode_transition(np.array(state[1]), self.source)
        return (tuple(s), tuple(s))
