import os
import sys
import getopt


def parse_command_line_options(print_options=False):
    optval = getopt.getopt(sys.argv[1:], 'n:d:fbs', [])
    itno = -1
    path = '.'
    falsify = False
    use_best_model = True
    simple_env = False
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
            use_best_model = False
    if itno != -1:
        path = os.path.join(path, 'run{}'.format(itno))
    flags = {'path': path,
             'simple': simple_env,
             'falsify': falsify,
             'use_best': use_best_model}
    if print_options:
        print('**** Command Line Options ****')
        for key in flags:
            print('{}: {}'.format(key, flags[key]))
    return flags
