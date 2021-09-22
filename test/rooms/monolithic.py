import os
import sys
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.train.single_mode import make_ars_model, learn_ars_model
from hybrid_gym.hybrid_env import HybridEnv
from hybrid_gym.selectors import UniformSelector, MaxJumpWrapper
from hybrid_gym.util.io import parse_command_line_options, save_log_info
from hybrid_gym.envs import make_rooms_model
from hybrid_gym.rl.ars import ARSParams, NNParams, NNPolicy


if __name__ == '__main__':

    flags = parse_command_line_options()
    automaton = make_rooms_model()
    modes = [mode for _, mode in automaton.modes.items()]
    selector = MaxJumpWrapper(UniformSelector(modes), 5)
    env = HybridEnv(automaton, selector, max_timesteps=125)

    if not os.path.exists(flags['path']):
        os.makedirs(flags['path'])

    nn_params = NNParams(2, 2, 1.0, 128)
    ars_params = ARSParams(50000, 30, 10, 0.03, 0.1, 0.95, 125)
    time_limits = {m: 25 for m in automaton.modes}
    ars_model = make_ars_model(ars_params, nn_params)

    _, log_info = ars_model.learn([env], True, eval_automaton=automaton,
                                  max_jumps=5, time_limits=time_limits)
    save_log_info(log_info, 'log', flags['path'])
