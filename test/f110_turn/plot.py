import os
import sys
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from matplotlib import pyplot as plt
from hybrid_gym.util.io import parse_command_line_options, plot_learning_curve, save_plot

ALGO_NAMES = ['ROSAC', 'AROSAC', 'NAIVE', 'DAGGER', 'MADDPG', 'PAIRED']
COLORS = ['red', 'green', 'pink', 'blue', 'cyan', 'grey']
PLOT_NAMES = ['samples', 'time', 'avg_jumps', 'mcts_avg_jumps', 'avg_prob', 'mcts_prob']


if __name__ == '__main__':
    flags = parse_command_line_options()
    num_runs = flags['abstract_samples']
    x_col = flags['ensemble']
    y_col = flags['gpu_num']
    min_x = int(1e9)
    plot_name = 'f110_' + PLOT_NAMES[y_col]

    for algo, color in zip(ALGO_NAMES, COLORS):
        path = os.path.join(flags['path'], algo)
        min_x = min(min_x, plot_learning_curve(
            'log', path, x_col, y_col, 1, num_runs+1, color, algo))

    plt.xlim(xmax=min_x, xmin=-100000.)
    save_plot(plot_name, flags['path'])
