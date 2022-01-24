import os
import sys
import argparse
import pathlib
from typing import List, Any
from multiprocessing import Process
from stable_baselines3 import HerReplayBuffer

sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.envs.pick_place.hybrid_env import make_pick_place_model
from hybrid_gym.synthesis.abstractions import Box, StateWrapper
from hybrid_gym.train.cegrl import cegrl
from hybrid_gym.hybrid_env import HybridAutomaton
from hybrid_gym.selectors import FixedSequenceSelector
from hybrid_gym.util.io import save_log_info
from hybrid_gym.envs.pick_place.mode import ModeType

sb2_hyperparams = dict(
    algo_name='sac', policy='MultiInputPolicy',
    gamma=0.95, buffer_size=1000000,
    batch_size=256, learning_rate=0.001,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy='future',
        online_sampling=True,
        max_episode_length=50,
    ),
    policy_kwargs=dict(
        net_arch=[64, 64],
    ),
    train_freq=1, learning_starts=1000,
)

sb3_hyperparams = dict(
    algo_name='tqc', policy='MultiInputPolicy',
    gamma=0.95, buffer_size=1000000,
    batch_size=2048, learning_rate=0.001,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy='future',
        online_sampling=True,
        max_episode_length=50,
    ),
    policy_kwargs=dict(
        net_arch=[512, 512, 512],
        n_critics=2,
    ),
    train_freq=1, learning_starts=1000,
)

class FalsifyFunc:
    '''
    Evaluation function used by the falsification algorithm.
    '''

    def __init__(self, mode):
        self.mode = mode

    def __call__(self, sass: List[Any]) -> float:
        rewards = [self.mode.reward(*sas) for sas in sass]
        return sum(rewards)


def run_cegrl(automaton: HybridAutomaton,
              num_objects: int,
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
    num_synth_iter = 15 if procedure == 'falsify' or procedure == 'cegrl' else 0
    #num_synth_iter = 2 if procedure == 'falsify' or procedure == 'cegrl' else 0

    pre = {}
    for m in automaton.modes:
        pre[m] = StateWrapper(automaton.modes[m], Box())
        for _ in range(100):
            pre[m].extend(automaton.modes[m].reset())

    time_limits = {m: 50 for m in automaton.modes}
    full_reset=(procedure == 'basic')
    mode_groups = [[automaton.modes[f'{mt.name}_{i}'] for i in range(num_objects)] for mt in ModeType]

    controllers, log_info = cegrl(
        automaton, pre, time_limits, mode_groups=mode_groups,
        num_iter=num_iter, num_synth_iter=num_synth_iter, abstract_synth_samples=0,
        num_falsification_iter=50,
        falsify_func=falsify_func,
        reward_offset=0.0,
        is_goal_env=True,
        sb3_train_kwargs=dict(
            total_timesteps=timesteps,
            is_goal_env=True,
            reward_offset=0.0,
        ),
        print_debug=False, verbose=verbose,
        use_best_model=False, save_path=save_path,
        init_check_train_timesteps=0,
        init_check_eval_episodes=1,
        dagger=(procedure == 'dagger'),
        full_reset=full_reset, plot_synthesized_regions=False,
        **sb2_hyperparams
    )

    # save the controllers
    for (mode_name, ctrl) in controllers.items():
        ctrl.save(os.path.join(save_path, mode_name + '.td3'))
    save_log_info(log_info, 'log', save_path)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='train controllers and mode predictor')
    ap.add_argument('--path', type=pathlib.Path, default='.',
                    help='directory in which models will be saved')
    ap.add_argument('--num-objects', type=int, default=3,
                    help='number of objects in the environment')
    ap.add_argument('--reward-type', choices=['dense', 'sparse'], default='sparse',
                    help='reward type')
    ap.add_argument('--timesteps', type=int, default=100000,
                    help='number of timesteps to train each controller in each iteration')
    ap.add_argument('--iter-scale', type=int, default=1,
                    help='multiplier for number of iterations for each procedure')
    ap.add_argument('--num-runs', type=int, default=5,
                    help='number of runs for each procedure')
    ap.add_argument('--best-model', action='store_true',
                    help='use this flag to evaluate the controller regularly and keep the best model')
    ap.add_argument('--all', action='store_true',
                    help='use this flag to train all procedures instead of specifying a list')
    ap.add_argument('--verbose', '-v', action='count', default=0,
                    help='each instance of -v increases verbosity')
    ap.add_argument('procedures', type=str, nargs='*',
                    help='procedures for which controllers will be trained')
    args = ap.parse_args()
    automaton = make_pick_place_model(num_objects=args.num_objects, reward_type=args.reward_type)
    procedure_list = ['basic', 'dagger', 'cegrl', 'falsify'] if args.all else args.procedures
    num_iter = dict(
        basic=5 * args.iter_scale,
        dagger=4 * args.iter_scale,
        synthesis=4 * args.iter_scale,
        falsify=3 * args.iter_scale,
    )

    for i in range(1, args.num_runs + 1):
        print(f'begin run {i}')
        for procedure in procedure_list:
            print(f'begin procedure {procedure}')
            run_cegrl(
                automaton=automaton,
                num_objects=args.num_objects,
                procedure=procedure,
                save_path=os.path.join(args.path, procedure, f'run{i}'),
                use_best_model=args.best_model,
                timesteps=args.timesteps,
                num_iter=num_iter[procedure],
                verbose=args.verbose,
            )
            print(f'end procedure {procedure}')
        print(f'end run {i}')
