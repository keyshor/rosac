from hybrid_gym.hybrid_env import HybridAutomaton
from hybrid_gym.envs.rooms.mode import RoomsMode, GridParams
from hybrid_gym.envs.rooms.transition import RoomsTrans
from matplotlib import pyplot as plt

DEFAULT_GRID_PARAMS = GridParams((8, 8), (2, 2), [(1, 3), (5, 7)], (3, 5))


def make_rooms_model(grid_params: GridParams = DEFAULT_GRID_PARAMS) -> HybridAutomaton:
    modes = [RoomsMode(grid_params, name) for name in ['left', 'right', 'up']]
    mode_dict = {m.name: m for m in modes}
    return HybridAutomaton(modes=modes, transitions=[
        RoomsTrans(source=m.name, modes=mode_dict) for m in modes])


# Simple tests
if __name__ == '__main__':

    # visualize the room
    DEFAULT_GRID_PARAMS.plot_room()
    plt.show()
