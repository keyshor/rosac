import numpy as np
from hybrid_gym.hybrid_env import HybridAutomaton
from hybrid_gym.envs.rooms.mode import RoomsMode, Direction
from hybrid_gym.envs.rooms.transition import RoomsTrans

def make_rooms_model(room_radius: float = 1.0,
                     door_radius: float = 0.5,
                     action_scale: float = 0.1,
                     ) -> HybridAutomaton:
    modes = [
        RoomsMode(
            name = name,
            direction = direction,
            room_radius = room_radius,
            door_radius = door_radius,
            action_scale = action_scale,
        )
        for (name, direction) in [
            ('rooms_up', Direction.UP),
            ('rooms_right', Direction.RIGHT),
            ('rooms_down', Direction.DOWN),
            ('rooms_left', Direction.LEFT),
        ]
    ]
    mode_dict = {m.name: m for m in modes}
    return HybridAutomaton(
        modes=modes,
        transitions=[
            RoomsTrans(source=m.name, modes=mode_dict)
            for m in modes
        ],
    )
