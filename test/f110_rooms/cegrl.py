import os
import sys
import argparse
import pathlib
from typing import List, Any
from multiprocessing import Process

sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.envs.f110_rooms.hybrid_env import make_f110_rooms_model
from hybrid_gym.synthesis.abstractions import Box, StateWrapper
from hybrid_gym.train.cegrl import cegrl
from hybrid_gym.hybrid_env import HybridEnv
from hybrid_gym.selectors import FixedSequenceSelector
from hybrid_gym.util.io import save_log_info

class FalsifyFunc:
    '''
    Evaluation function used by the falsification algorithm.
    '''

    def __init__(self, mode):
        self.mode = mode

    def __call__(self, sass: List[Any]) -> float:
        rewards = [self.mode.reward(*sas) for sas in sass]
        return sum(rewards)


def run_f110(automaton: HybridEnv,
             procedure: str,
             save_path: str,
             use_best_model: bool,
             timesteps: int,
             num_iter: int,
             verbose: int,
             ) -> None:
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    falsify_func = {name: FalsifyFunc(mode) for name, mode in automaton.modes.items()} \
        if procedure == 'falsify' else None
    num_synth_iter = 10 if procedure == 'falsify' or procedure == 'dagger' else 0

    pre = {}
    for m in automaton.modes:
        pre[m] = StateWrapper(automaton.modes[m], Box())
        for _ in range(10):
            pre[m].extend(automaton.modes[m].reset())

    time_limits = {m: 50 for m in automaton.modes}

    controllers, log_info = cegrl(
        automaton, pre, time_limits,
        algo_name='td3', policy='MlpPolicy',
        num_iter=num_iter, num_synth_iter=num_synth_iter, abstract_synth_samples=0,
        falsify_func=falsify_func,
        print_debug=False, verbose=verbose,
        batch_size=100,
        action_noise_scale=4.0,
        learning_rate=0.0003,
        buffer_size=50000,
        policy_kwargs=dict(net_arch=[200, 50]),
        sb3_train_kwargs=dict(total_timesteps=timesteps),
        init_check_min_episode_length=10,
        use_best_model=use_best_model, save_path=save_path,
        full_reset=True, plot_synthesized_regions=True)

    # save the controllers
    for (mode_name, ctrl) in controllers.items():
        ctrl.save(os.path.join(save_path, mode_name + '.td3'))
    save_log_info(log_info, 'log', save_path)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='train controllers and mode predictor')
    ap.add_argument('--path', type=pathlib.Path, default='.',
                    help='directory in which models will be saved')
    ap.add_argument('--no-throttle', action='store_true',
                    help='use this flag to disable throttle in the environment')
    ap.add_argument('--timesteps', type=int, default=20000,
                    help='number of timesteps to train each controller in each iteration')
    ap.add_argument('--iter-scale', type=int, default=2,
                    help='multiplier for number of iterations for each procedure')
    ap.add_argument('--num-runs', type=int, default=5,
                    help='number of runs for each procedure')
    ap.add_argument('--good-start-region', action='store_true',
                    help='use this flag to give the training algorithm a bad start region')
    ap.add_argument('--best-model', action='store_true',
                    help='use this flag to evaluate the controller regularly and keep the best model')
    ap.add_argument('--all', action='store_true',
                    help='use this flag to train all procedures instead of specifying a list')
    ap.add_argument('--verbose', '-v', action='count', default=0,
                    help='each instance of -v increases verbosity')
    ap.add_argument('procedures', type=str, nargs='*',
                    help='procedures for which controllers will be trained')
    args = ap.parse_args()
    automaton = make_f110_rooms_model(use_throttle=not args.no_throttle, observe_heading=True, bad_start_region=not args.good_start_region)
    procedure_list = ['basic', 'dagger', 'falsify'] if args.all else args.procedures
    num_iter = dict(
        basic=5 * args.iter_scale,
        dagger=3 * args.iter_scale,
        falsify=2 * args.iter_scale,
    )

    for i in range(1, args.num_runs + 1):
        print(f'begin run {i}')
        for procedure in procedure_list:
            print(f'begin procedure {procedure}')
            run_f110(
                automaton=automaton,
                procedure=procedure,
                save_path=os.path.join(args.path, procedure, f'run{i}'),
                use_best_model=args.best_model,
                timesteps=args.timesteps,
                num_iter=num_iter[procedure],
                verbose=args.verbose,
            )
            print(f'end procedure {procedure}')
        print(f'end run {i}')
