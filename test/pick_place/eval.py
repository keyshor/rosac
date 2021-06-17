import os
import sys
import gym
from stable_baselines import HER
sys.path.append(os.path.join('..', '..'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.train.single_mode import BaselineCtrlWrapper, GymGoalEnvWrapper
from hybrid_gym.envs import make_pick_place_model
from hybrid_gym.hybrid_env import HybridGoalEnv
from hybrid_gym.selectors import FixedSequenceSelector

if __name__ == '__main__':
    num_objects = 3
    automaton = make_pick_place_model(num_objects=num_objects)

    env = HybridGoalEnv(
        automaton=automaton,
        selector=FixedSequenceSelector(
            mode_list=[
                automaton.modes['ModeType.MOVE_WITHOUT_OBJ'],
                automaton.modes['ModeType.PICK_OBJ'],
                automaton.modes['ModeType.MOVE_WITH_OBJ'],
                automaton.modes['ModeType.PLACE_OBJ_PT1'],
                automaton.modes['ModeType.PLACE_OBJ_PT2'],
            ] * num_objects
        )
    )
    # controllers = {name: BaselineCtrlWrapper.load(f'{name}.her', algo_name='her', env=env)
    #               for name in automaton.modes}

    # for _ in range(20):
    #    observation = env.reset()
    #    mode = ''
    #    e = 0
    #    done = False
    #    while not done:
    #        e += 1
    #        #mode = mode_predictor.get_mode(observation)
    #        if mode != env.mode.name:
    #            mode = env.mode.name
    #            print(f'switched to {mode}')
    #        delta = controllers[mode].get_action(observation)
    #        observation, reward, done, info = env.step(delta)
    #    if env.mode.is_safe(env.state):
    #        print(f'terminated normally after {e} steps')
    #    else:
    #        print(f'safety violation after {e} steps')
    #for (name, mode) in [('ModeType.PLACE_OBJ_PT2', automaton.modes['ModeType.PLACE_OBJ_PT2'])]:
    for name, mode in automaton.modes.items():
        goal_env = GymGoalEnvWrapper(mode, automaton.transitions[name])
        ctrl = BaselineCtrlWrapper.load(f'{name}.her', algo_name='her', env=goal_env)
        print(f'evaluating on mode {name}')
        for _ in range(5):
            obs = goal_env.reset()
            done = False
            for _ in range(100):
                if not done:
                    action = ctrl.get_action(obs)
                    obs, _, done, _ = goal_env.step(action)
                goal_env.render()
    #env = gym.envs.robotics.fetch.pick_and_place.FetchPickAndPlaceEnv()
    #model = HER.load('FetchPickAndPlace-v1.zip', env=env)
    #for _ in range(5):
    #    obs = env.reset()
    #    done = False
    #    for _ in range(100):
    #        if not done:
    #            action, _ = model.predict(obs)
    #            obs, _, done, _ = env.step(action)
    #        env.render()
