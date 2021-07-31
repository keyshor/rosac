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

# env = HybridEnv(
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
    # flatten_obs=True
)
controllers: dict = {}

trials_per_mode = 100
steps_per_trial = 100
end_to_end_trials = 100
steps_per_mode = 100


def eval_single(name):
    mode = automaton.modes[name]
    #mode_env = GymEnvWrapper(mode, automaton.transitions[name], flatten_obs=True)
    mode_env = GymGoalEnvWrapper(mode, automaton.transitions[name])
    ctrl = controllers[name]
    num_successes = 0
    for _ in range(trials_per_mode):
        obs = mode_env.reset()
        done = False
        for _ in range(steps_per_trial):
            if not done:
                action = ctrl.get_action(obs)
                obs, _, done, _ = mode_env.step(action)
            # mode_env.render()
        if mode.is_success(mode_env.state):
            num_successes += 1
    print(f'success rate for mode {name} is {num_successes}/{trials_per_mode}')


def eval_end_to_end():
    num_successes = 0
    for _ in range(end_to_end_trials):
        observation = env.reset()
        steps_in_cur_mode = 0
        mode = ''
        done = False
        while not done:
            if mode != env.mode.name:
                mode = env.mode.name
                steps_in_cur_mode = 0
                #print(f'switched to {mode}')
            delta = controllers[mode].get_action(observation)
            observation, reward, done, info = env.step(delta)
            # env.render()
            steps_in_cur_mode += 1
            if steps_in_cur_mode > steps_per_mode:
                #print(f'stuck in mode {mode} for {steps_per_mode} steps')
                break
        if env.selector.index == len(env.selector.mode_list) - 1 \
                and env.mode.is_success(env.state):
            num_successes += 1
    print(f'end-to-end success rate is {num_successes}/{end_to_end_trials}')


if __name__ == '__main__':
    save_path = '.'
    if len(sys.argv) >= 2:
        save_path = sys.argv[1]
        mode_list = sys.argv[2:]
    else:
        mode_list = list(automaton.modes)
    for name in mode_list:
        controllers[name] = BaselineCtrlWrapper.load(
            os.path.join(save_path, name, 'best_model.zip'),
            algo_name='her',
            env=env,
        )
        #controllers[name] = SpectrlCtrlWrapper.load(f'{name}.spectrl')
        eval_single(name)
    if len(sys.argv) <= 1:
        eval_end_to_end()
