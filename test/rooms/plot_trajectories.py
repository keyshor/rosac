import os
import sys
import argparse
import pathlib
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.util.wrappers import BaselineCtrlWrapper, GymEnvWrapper
from hybrid_gym.train.mode_pred import ScipyModePredictor
from hybrid_gym.envs.rooms.hybrid_env import make_rooms_model
from hybrid_gym.hybrid_env import HybridEnv
from hybrid_gym.selectors import UniformSelector, MaxJumpWrapper

import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_walls(mode, ax):
    hfs = 0.5 * mode.grid_params.full_size
    right_top_outside = mode.grid_params.full_size - hfs
    right_top_inside = mode.grid_params.partition_size - hfs
    left_bottom_inside = mode.grid_params.wall_size - hfs
    left_bottom_outside = -hfs
    vdoor_bounds = mode.grid_params.vdoor - hfs[1]
    hdoor_bounds = mode.grid_params.hdoor - hfs[0]
    right_outside = right_top_outside[0]
    top_outside = right_top_outside[1]
    right_inside = right_top_inside[0]
    top_inside = right_top_inside[1]
    left_inside = left_bottom_inside[0]
    bottom_inside = left_bottom_inside[1]
    left_outside = left_bottom_outside[0]
    bottom_outside = left_bottom_outside[1]
    vdoor_bottom = vdoor_bounds[0]
    vdoor_top = vdoor_bounds[1]
    hdoor_left = hdoor_bounds[0]
    hdoor_right = hdoor_bounds[1]

    # upper left
    ax.add_line(mpl.lines.Line2D(
        [left_outside, hdoor_left, hdoor_left, left_inside, left_inside, left_outside, left_outside],
        [top_outside, top_outside, top_inside, top_inside, vdoor_top, vdoor_top, top_outside],
    ))
    # upper right
    ax.add_line(mpl.lines.Line2D(
        [right_outside, right_outside, right_inside, right_inside, hdoor_right, hdoor_right, right_outside],
        [top_outside, vdoor_top, vdoor_top, top_inside, top_inside, top_outside, top_outside],
    ))
    # lower right
    ax.add_line(mpl.lines.Line2D(
        [right_outside, hdoor_right, hdoor_right, right_inside, right_inside, right_outside, right_outside],
        [bottom_outside, bottom_outside, bottom_inside, bottom_inside, vdoor_bottom, vdoor_bottom, bottom_outside],
    ))
    # lower left
    ax.add_line(mpl.lines.Line2D(
        [left_outside, left_outside, left_inside, left_inside, hdoor_left, hdoor_left, left_outside],
        [bottom_outside, vdoor_bottom, vdoor_bottom, bottom_inside, bottom_inside, bottom_outside, bottom_outside],
    ))

def eval_single(automaton, name, save_path):
    mode = automaton.modes[name]
    env = GymEnvWrapper(mode, automaton.transitions[name])
    controller = BaselineCtrlWrapper.load(
        os.path.join(save_path, name, 'best_model.zip'),
        algo_name='td3',
        env=env,
    )
    state_history = []
    start_history = []
    end_history = []
    nonterm = 0
    normal = 0
    crash = 0
    for _ in range(100):
        observation = env.reset()
        e = 0
        done = False
        start_history.append(env.state)
        while not done:
            if e > 100:
                break
            e += 1
            action = controller.get_action(observation)
            state_history.append(env.state)
            observation, reward, done, info = env.step(action)
            #print(reward)
        end_history.append(env.state)
        if e > 50:
            #print('spent more than 50 steps in the mode')
            nonterm += 1
        elif env.mode.is_safe(env.state):
            #print(f'terminated normally after {e} steps')
            normal += 1
        else:
            #print(f'crash after {e} steps')
            crash += 1
    fig, ax = plt.subplots()
    plot_walls(mode=mode, ax=ax)
    x_hist = [s[1][0] for s in state_history]
    y_hist = [s[1][1] for s in state_history]
    x_start = [s[1][0] for s in start_history]
    y_start = [s[1][1] for s in start_history]
    x_end = [s[1][0] for s in end_history]
    y_end = [s[1][1] for s in end_history]
    ax.scatter(x_hist, y_hist, s=1, label='trajectory')
    ax.scatter(x_start, y_start, s=1, label='start')
    ax.scatter(x_end, y_end, s=1, label='end')
    ax.legend(markerscale=10)
    ax.set_title(name)
    ax.set_aspect('equal')
    fig.savefig(f'trajectories_{name}.png')
    print(f'{name}: nonterm = {nonterm}, normal = {normal}, crash = {crash}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='evaluate controllers and mode predictor')
    ap.add_argument('--path', type=pathlib.Path, default='.',
                    help='directory in which models are stored')
    ap.add_argument('--all', action='store_true',
                    help='use this flag to evaluate instead of specifying a list')
    ap.add_argument('--mode_length', type=int, default=100,
                    help='maximum number of time-steps in each mode')
    ap.add_argument('modes', type=str, nargs='*',
                    help='modes for which controllers will be evaluated')
    args = ap.parse_args()
    automaton = make_rooms_model()

    mode_list = list(automaton.modes) if args.all else args.modes
    for name in mode_list:
        eval_single(automaton, name, args.path)
