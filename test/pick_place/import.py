import os
import sys
sys.path.append(os.path.join('..', '..'))  # nopep8

from hybrid_gym.envs.pick_place import PickPlaceMode, ModeType

if __name__ == '__main__':
    for mt in ModeType:
        print(f'mode_type = {mt}')
        m = PickPlaceMode(mode_type=mt, num_objects=3)
        for _ in range(5):
            s = m.reset()
            for _ in range(60):
                m.render(s)
