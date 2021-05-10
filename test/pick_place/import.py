import os
import sys
sys.path.append(os.path.join('..', '..'))  # nopep8

from hybrid_gym.envs import make_pick_place_model

if __name__ == '__main__':
    pick_place_automaton = make_pick_place_model(num_objects=3)
    for (name, m) in pick_place_automaton.modes.items():
        print(f'mode name = {name}')
        print(m.observation_space)
        for _ in range(5):
            s = m.reset()
            #for _ in range(60):
            #    m.render(s)
