import os
import sys

sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.envs import make_f110_model
from hybrid_gym.synthesis.abstractions import Box, StateWrapper
from hybrid_gym.train.cegrl import cegrl
from hybrid_gym.hybrid_env import HybridGoalEnv
from hybrid_gym.selectors import FixedSequenceSelector


if __name__ == '__main__':
    automaton = make_f110_model(straight_lengths=[10], use_throttle=True)

    use_best_model = 0
    save_path = '.'
    if len(sys.argv) > 1:
        save_path = sys.argv[1]
    if len(sys.argv) > 2:
        use_best_model = int(sys.argv[2])

    pre = {}
    for m in automaton.modes:
        pre[m] = StateWrapper(automaton.modes[m], Box())
        for _ in range(10):
            pre[m].extend(automaton.modes[m].reset())

    time_limits = {m: 50 for m in automaton.modes}

    controllers, log_info = cegrl(
        automaton, pre, time_limits,
        algo_name='td3', policy='MlpPolicy',
        num_iter=3, num_synth_iter=3, n_synth_samples=5, abstract_synth_samples=0,
        num_falsification_iter=3, num_falsification_samples=5, num_falsification_top_samples=5,
        print_debug=True, verbose=0,
        batch_size=100,
        action_noise_scale=4.0,
        learning_rate=0.0003,
        buffer_size=50000,
        policy_kwargs=dict(net_arch=[200, 50]),
        sb3_train_kwargs=dict(total_timesteps=200),
        use_best_model=use_best_model, save_path=save_path)

    # save the controllers
    for (mode_name, ctrl) in controllers.items():
        ctrl.save(os.path.join(save_path, mode_name + '.td3'))
