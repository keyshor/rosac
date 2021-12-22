import os
import sys
import getopt
import numpy as np

from typing import Optional
from matplotlib import pyplot as plt


def parse_command_line_options(print_options=False):
    optval = getopt.getopt(sys.argv[1:], 'n:d:a:v:m:fbstrgci', [])
    itno = -1
    path = '.'
    falsify = False
    use_best = False
    synthesize = False
    no_throttle = False
    render = False
    inductive_ce = False
    abstract_samples = 0
    use_gpu = False
    dagger = False
    mode = ''
    gpu_num = 0
    for option in optval[0]:
        if option[0] == '-n':
            itno = int(option[1])
        if option[0] == '-v':
            gpu_num = int(option[1])
        if option[0] == '-d':
            path = option[1]
        if option[0] == '-s':
            synthesize = True
        if option[0] == '-c':
            dagger = True
        if option[0] == '-f':
            falsify = True
        if option[0] == '-b':
            use_best = True
        if option[0] == '-t':
            no_throttle = True
        if option[0] == '-r':
            render = True
        if option[0] == '-a':
            abstract_samples = int(option[1])
        if option[0] == '-g':
            use_gpu = True
        if option[0] == '-m':
            mode = option[1]
        if option[0] == '-i':
            inductive_ce = True
    if itno != -1:
        path = os.path.join(path, 'run{}'.format(itno))
    flags = {'path': path,
             'gpu_num': gpu_num,
             'synthesize': synthesize,
             'falsify': falsify,
             'best': use_best,
             'no_throttle': no_throttle,
             'render': render,
             'abstract_samples': abstract_samples,
             'gpu': use_gpu,
             'dagger': dagger,
             'mode': mode,
             'inductive_ce': inductive_ce}
    if print_options:
        print('**** Command Line Options ****')
        for key in flags:
            print('{}: {}'.format(key, flags[key]))
    return flags


def save_log_info(log_info: np.ndarray, name: str, path: str):
    path = os.path.join(path, name + '.npy')
    np.save(path, log_info)


def load_log_info(name: str, path: str, itno: Optional[int] = None) -> np.ndarray:
    if itno is not None:
        path = os.path.join(path, 'run{}'.format(itno))
    path = os.path.join(path, name + '.npy')
    return np.load(path)


def plot_error_bar(x: np.ndarray, data: np.ndarray, color: str, label: str, points: bool = False):
    '''
    Plot the error bar from the data.
    Parameters:
        samples_per_iter - number of sample rollouts per iteration of the algorithm
        data - (3+)-tuple of np.array (curve, lower error bar, upper error bar, ...)
        color - color of the plot
    '''
    plt.subplots_adjust(bottom=0.126)
    plt.rcParams.update({'font.size': 18})
    if points:
        plt.errorbar(x, data[0], data[0]-data[1], fmt='--o', color=color, label=label)
    else:
        plt.plot(x, data[0], color=color, label=label)
        plt.fill_between(x, data[1], data[2], color=color, alpha=0.15)


def extract_plot_data(name: str, path: str, column_num: int, low: int, up: int):
    '''
    Load and parse log_info to generate error bars
    Parameters:
        column_num - column number in log.npy to use
        low - lower limit on run number
        up - upper limit on run number
    Returns:
        3-tuple of numpy arrays (curve, lower error bar, upper error bar)
    '''
    log_infos = []
    min_length = 10000000
    for itno in range(low, up):
        log_info = np.transpose(load_log_info(name, path, itno))[column_num]
        log_info = np.append([0], log_info)
        min_length = min(min_length, len(log_info))
        log_infos.append(log_info)
    log_infos = [log_info[:min_length] for log_info in log_infos]
    data = np.array(log_infos)
    curve = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return curve, (curve - std), (curve + std)


def save_plot(name, path, show=False, scientific=True):
    '''
    Save and render current plot
    '''
    plt.rcParams.update({'font.size': 14})
    plt.legend()
    ax = plt.gca()
    ax.xaxis.major.formatter._useMathText = True
    if scientific:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    path = os.path.join(path, name + '.pdf')
    plt.savefig(path, format='pdf')
    if show:
        plt.show()


def plot_learning_curve(name: str, path: str, y_col: int, low: int, up: int,
                        color: str, label: str, points: bool = False):
    x = extract_plot_data(name, path, 0, low, up)[0]
    data = extract_plot_data(name, path, y_col, low, up)
    plot_error_bar(x, data, color, label, points=points)
    return x[-1]
