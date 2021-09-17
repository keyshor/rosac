import os
import sys
import getopt

import numpy as np


def parse_command_line_options(print_options=False):
    optval = getopt.getopt(sys.argv[1:], 'n:d:a:fbstrg', [])
    itno = -1
    path = '.'
    falsify = False
    use_best = False
    simple_env = False
    no_throttle = False
    render = False
    abstract_samples = 0
    use_gpu = False
    for option in optval[0]:
        if option[0] == '-n':
            itno = int(option[1])
        if option[0] == '-d':
            path = option[1]
        if option[0] == '-s':
            simple_env = True
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
    if itno != -1:
        path = os.path.join(path, 'run{}'.format(itno))
    flags = {'path': path,
             'simple': simple_env,
             'falsify': falsify,
             'best': use_best,
             'no_throttle': no_throttle,
             'render': render,
             'abstract_samples': abstract_samples,
             'gpu': use_gpu}
    if print_options:
        print('**** Command Line Options ****')
        for key in flags:
            print('{}: {}'.format(key, flags[key]))
    return flags


def save_log_info(log_info, name, path):
    path = os.path.join(path, name)
    np.save(path, log_info)
