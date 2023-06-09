import os
import sys
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from matplotlib import pyplot as plt
from hybrid_gym.util.io import parse_command_line_options, plot_learning_curve, save_plot

COLOR = 'blue'
PLOT_NAMES = ['samples', 'time', 'avg_jumps', 'mcts_avg_jumps', 'avg_prob', 'mcts_prob']


if __name__ == '__main__':
    flags = parse_command_line_options()
    num_runs = flags['abstract_samples']
    x_col = flags['ensemble']
    y_col = flags['gpu_num']
    min_x = int(1e9)
    plot_name = PLOT_NAMES[y_col] + '_' + PLOT_NAMES[x_col]

    min_x = min(min_x, plot_learning_curve(
        'log', flags['path'], x_col, y_col, 1, num_runs+1, COLOR, 'learning_curve'))

    plt.xlim(xmax=min_x)
    save_plot(plot_name, flags['path'])
