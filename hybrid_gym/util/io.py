import os
import sys
import getopt


def parse_command_line_options(print_options=False):
    optval = getopt.getopt(sys.argv[1:], 'n:d:fbst', [])
    itno = -1
    path = '.'
    falsify = False
    no_best_model = False
    simple_env = False
    no_throttle = False
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
            no_best_model = True
        if option[0] == '-t':
            no_throttle = True
    if itno != -1:
        path = os.path.join(path, 'run{}'.format(itno))
    flags = {'path': path,
             'simple': simple_env,
             'falsify': falsify,
             'no_best': no_best_model,
             'no_throttle': no_throttle}
    if print_options:
        print('**** Command Line Options ****')
        for key in flags:
            print('{}: {}'.format(key, flags[key]))
    return flags
