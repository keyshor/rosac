import os
import sys
import argparse
import pathlib
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

from hybrid_gym.envs.rooms.hybrid_env import make_rooms_model

if __name__ == '__main__':
    automaton = make_rooms_model()
    ap = argparse.ArgumentParser(description='train controllers and mode predictor')
    ap.add_argument('mode', choices=list(automaton.modes),
                    help='mode to debug')
    args = ap.parse_args()

    m = automaton.modes[args.mode]
    st = m.reset()
    m.render(st)
