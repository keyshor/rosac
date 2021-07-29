import os
import sys

sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.envs import make_pick_place_model
from hybrid_gym.synthesis.abstractions import Box, StateWrapper
from hybrid_gym.train.cegrl import cegrl


if __name__ == '__main__':
    automaton = make_pick_place_model(num_objects=3)

    pre = {}
    for m in automaton.modes:
        pre[m] = StateWrapper(automaton.modes[m], Box())
        for _ in range(100):
            pre[m].extend(automaton.modes[m].reset())

    time_limits = {m: 50 for m in automaton.modes}

    controllers = cegrl(automaton, pre, time_limits, algo_name='her', steps_per_iter=100000,
                        num_iter=10, num_synth_iter=10, abstract_samples=0, print_debug=True,
                        wrapped_algo='sac', verbose=2, gamma=0.95, buffer_size=1000000,
                        ent_coef='auto', goal_selection_strategy='future',
                        n_sampled_goal=4, train_freq=1, learning_starts=1000)

    # save the controllers
    for (mode_name, ctrl) in controllers.items():
        ctrl.save(f'{mode_name}.her')
