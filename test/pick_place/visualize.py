import os
import sys
import gym
from stable_baselines.her import HERGoalEnvWrapper
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.util.wrappers import BaselineCtrlWrapper, SpectrlCtrlWrapper, GymEnvWrapper, GymGoalEnvWrapper
from hybrid_gym.envs import make_pick_place_model
from hybrid_gym.hybrid_env import HybridEnv, HybridGoalEnv
from hybrid_gym.selectors import FixedSequenceSelector

num_objects = 3
automaton = make_pick_place_model(num_objects=num_objects,
                                  reward_type='sparse',
                                  distance_threshold=0.015)

#env = HybridEnv(
env = HybridGoalEnv(
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
    #flatten_obs=True
)
controllers = {}

num_trials = 5
steps_per_trial = 100

def visualize_mode(name):
    mode = automaton.modes[name]
    #mode_env = GymEnvWrapper(mode, automaton.transitions[name], flatten_obs=True)
    mode_env = GymGoalEnvWrapper(mode, automaton.transitions[name])
    ctrl = controllers[name]
    num_successes = 0
    for _ in range(num_trials):
        obs = mode_env.reset()
        done = False
        for _ in range(steps_per_trial):
            #if not done:
            #    action = ctrl.get_action(obs)
            #    obs, _, done, _ = mode_env.step(action)
            mode_env.render()
        if mode.is_success(mode_env.state):
            num_successes += 1
    #print(f'success rate for mode {name} is {num_successes}/{num_trials}')

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        mode_list = sys.argv[1:]
    else:
        mode_list = list(automaton.modes)
    for name in mode_list:
        controllers[name] = BaselineCtrlWrapper.load(
            os.path.join(name, 'best_model.zip'),
            algo_name='her',
            env=env,
        )
        #controllers[name] = SpectrlCtrlWrapper.load(f'{name}.spectrl')
        visualize_mode(name)
