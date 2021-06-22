import os
import sys
sys.path.append(os.path.join('..', '..'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.envs import make_pick_place_model
from hybrid_gym.util.wrappers import GymGoalEnvWrapper

if __name__ == '__main__':
    pick_place_automaton = make_pick_place_model(num_objects=3)
    # for (name, m) in pick_place_automaton.modes.items():
    #    goal_env = GymGoalEnvWrapper(
    #        m, pick_place_automaton.transitions[name], None, None
    #    )
    #    print(f'mode name = {name}')
    #    #print(m.observation_space)
    #    print(goal_env.observation_space)
    #    for _ in range(5):
    #        #s = m.reset()
    #        #m.multi_obj.reset()
    #        goal_env.reset()
    #        for _ in range(60):
    #            #m.render(s)
    #            #m.multi_obj.render()
    #            goal_env.render()
    #name = 'ModeType.MOVE_WITHOUT_OBJ'
    #m = pick_place_automaton.modes[name]
    # for _ in range(5):
    #    s = m.end_to_end_reset()
    #    for _ in range(60):
    #        m.render(s)
