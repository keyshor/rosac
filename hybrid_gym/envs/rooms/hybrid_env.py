from hybrid_gym.hybrid_env import HybridAutomaton
from hybrid_gym.envs.rooms.mode import RoomsMode, GridParams
from hybrid_gym.envs.rooms.transition import RoomsTrans

DEFAULT_GRID_PARAMS = GridParams((8, 8), (2, 2), (3.5, 4.5), (3.5, 4.5))


def make_rooms_model(grid_params: GridParams = DEFAULT_GRID_PARAMS) -> HybridAutomaton:
    modes = [RoomsMode(grid_params, name)
             for name in ['left', 'right', 'up', 'down']]
    mode_dict = {m.name: m for m in modes}
    return HybridAutomaton(
        modes=modes,
        transitions=[
            RoomsTrans(source=m.name, modes=mode_dict)
            for m in modes
        ],
    )
