import os
import sys
from typing import List, Any
from multiprocessing import Process

sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.envs import make_f110_model
from hybrid_gym.synthesis.abstractions import Box, StateWrapper
from hybrid_gym.train.cegrl import cegrl
from hybrid_gym.hybrid_env import HybridGoalEnv
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


def run_f110(procedure: str, save_path: str, use_best_model: bool) -> None:
    automaton = make_f110_model(straight_lengths=[10], use_throttle=True)

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
        num_iter=5, num_synth_iter=num_synth_iter, abstract_synth_samples=0,
        falsify_func=falsify_func,
        print_debug=False, verbose=0,
        batch_size=100,
        action_noise_scale=4.0,
        learning_rate=0.0003,
        buffer_size=50000,
        policy_kwargs=dict(net_arch=[200, 50]),
        sb3_train_kwargs=dict(total_timesteps=200),
        init_check_min_episode_length=10,
        use_best_model=use_best_model, save_path=save_path)

    # save the controllers
    for (mode_name, ctrl) in controllers.items():
        ctrl.save(os.path.join(save_path, mode_name + '.td3'))
    save_log_info(log_info, 'log', save_path)

if __name__ == '__main__':
    #for i in range(1, 6):
    #    print(f'begin run {i}')
    #    processes = [
    #        Process(target=run_f110, kwargs=dict(
    #            procedure=procedure,
    #            save_path=os.path.join('cegrl_models', procedure, f'run{i}'),
    #            use_best_model=False,
    #        ))
    #        for procedure in ['basic', 'dagger', 'falsify']
    #    ]
    #    for p in processes:
    #        p.start()
    #    for p in processes:
    #        p.join()
    #    print(f'end run {i}')
    #for i in range(1, 6):
    for i in range(1, 2):
        print(f'begin run {i}')
        #for procedure in ['basic', 'dagger', 'falsify']:
        for procedure in ['basic']:
            print(f'begin procedure {procedure}')
            run_f110(
                procedure=procedure,
                save_path=os.path.join('cegrl_models', procedure, f'run{i}'),
                use_best_model=False,
            )
            print(f'end procedure {procedure}')
        print(f'end run {i}')
