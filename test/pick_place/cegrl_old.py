import os
import sys
from stable_baselines3 import HerReplayBuffer

sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.envs import make_pick_place_model
from hybrid_gym.synthesis.abstractions import Box, StateWrapper
from hybrid_gym.train.cegrl import cegrl
from hybrid_gym.hybrid_env import HybridGoalEnv
from hybrid_gym.selectors import FixedSequenceSelector
from hybrid_gym.envs.pick_place.mode import ModeType

class FalsifyFunc:
    '''
    Evaluation function used by the falsification algorithm.
    '''

    def __init__(self, mode):
        self.mode = mode

    def __call__(self, sass: List[Any]) -> float:
        rewards = [self.mode.reward(*sas) for sas in sass]
        return sum(rewards)

if __name__ == '__main__':
    num_objects = 3
    automaton = make_pick_place_model(num_objects=num_objects, reward_type='sparse')

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

    mode_groups = [[automaton.modes[f'{mt.name}_{i}'] for i in range(num_objects)] for mt in ModeType]

    controllers = cegrl(automaton, pre, time_limits, mode_groups=mode_groups,
                        algo_name='tqc', policy='MultiInputPolicy',
                        num_iter=10, num_synth_iter=10, abstract_synth_samples=0, print_debug=True,
                        verbose=0, gamma=0.95, buffer_size=1000000,
                        batch_size=2048, learning_rate=0.001,
                        reward_offset=0.0,
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
                        is_goal_env=True,
                        sb3_train_kwargs=dict(
                            total_timesteps=200,
                            is_goal_env=True,
                            reward_offset=0.0,
                        ),
                        train_freq=1, learning_starts=1000,
                        use_best_model=use_best_model, save_path=save_path),

    # save the controllers
    for (mode_name, ctrl) in controllers.items():
        ctrl.save(os.path.join(save_path, mode_name + '.her'))
