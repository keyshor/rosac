import os
import sys
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from matplotlib import pyplot as plt
from hybrid_gym.util.io import parse_command_line_options, plot_learning_curve, save_plot

COLOR = 'blue'
NUM_RUNS = 5


if __name__ == '__main__':
    flags = parse_command_line_options()
    min_x = int(1e9)
    y_col = -2
    plot_name = 'avg_case_plot'
    if flags['falsify']:
        y_col = -1
        plot_name = 'worst_case_plot'

    min_x = min(min_x, plot_learning_curve(
        'log', flags['path'], y_col, 1, NUM_RUNS+1, COLOR, 'learning_curve'))

    plt.xlim(xmax=min_x)
    save_plot(plot_name, flags['path'])
