import os
import sys

sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.envs import make_pick_place_model
from hybrid_gym.synthesis.abstractions import Box, StateWrapper
from hybrid_gym.train.cegrl import cegrl
from hybrid_gym.hybrid_env import HybridGoalEnv
from hybrid_gym.selectors import FixedSequenceSelector


if __name__ == '__main__':
    num_objects = 3
    automaton = make_pick_place_model(num_objects=num_objects)
    reload_env = HybridGoalEnv(
        automaton=automaton,
        selector=FixedSequenceSelector(
            mode_list=[
                automaton.modes['ModeType.MOVE_WITHOUT_OBJ'],
                automaton.modes['ModeType.PICK_OBJ_PT1'],
                automaton.modes['ModeType.PICK_OBJ_PT2'],
                automaton.modes['ModeType.PICK_OBJ_PT3'],
                automaton.modes['ModeType.MOVE_WITH_OBJ'],
                automaton.modes['ModeType.PLACE_OBJ_PT1'],
                automaton.modes['ModeType.PLACE_OBJ_PT2'],
            ] * num_objects
        ),
    )

    use_best_model = 0
    save_path = '.'
    if len(sys.argv) > 1:
        save_path = sys.argv[1]
    if len(sys.argv) > 2:
        use_best_model = int(sys.argv[2])

    pre = {}
    for m in automaton.modes:
        pre[m] = StateWrapper(automaton.modes[m], Box())
        for _ in range(100):
            pre[m].extend(automaton.modes[m].reset())

    time_limits = {m: 50 for m in automaton.modes}

    controllers = cegrl(automaton, pre, time_limits, reload_env, algo_name='her', steps_per_iter=10,
                        num_iter=10, num_synth_iter=10, abstract_samples=0, print_debug=True,
                        wrapped_algo='sac', verbose=2, gamma=0.95, buffer_size=1000000,
                        ent_coef='auto', goal_selection_strategy='future',
                        n_sampled_goal=4, train_freq=1, learning_starts=1000,
                        use_best_model=use_best_model, save_path=save_path)

    # save the controllers
    for (mode_name, ctrl) in controllers.items():
        ctrl.save(os.path.join(save_path, mode_name + '.her'))
