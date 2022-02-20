from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Dict, Optional
from sklearn import svm

from hybrid_gym.model import StateType, Mode, Controller, Transition
from hybrid_gym.hybrid_env import HybridAutomaton
from hybrid_gym.util.test import get_rollout

import numpy as np
import time


class RewardFunc(metaclass=ABCMeta):

    mode: Optional[Mode]
    automaton: Optional[HybridAutomaton]

    def __init__(self, mode: Mode, automaton: HybridAutomaton):
        self.mode = mode
        self.automaton = automaton
        self.mname = mode.name

    def __call__(self, state: StateType, action: np.ndarray,
                 next_state: StateType) -> float:
        '''
        Returns the reward for the given step.
        '''
        assert self.mode is not None
        return self.mode.reward(state, action, next_state)

    @abstractmethod
    def obs_reward(self, obs: np.ndarray, action, next_obs: np.ndarray,
                   orig_reward: float, is_success: bool,
                   jump_obs: Optional[List[Tuple[str, np.ndarray]]] = None) -> float:
        '''
        Return reward for a step corresponding to the given observed step.
        orig_reward is the reward assigned by the environment to the step.
        is_success denotes if this step lead to a transition out of self.mode.
        '''
        pass

    @abstractmethod
    def update(self, collected_states: List[Tuple[StateType, StateType, Optional[Transition]]],
               controllers: Dict[str, List[Controller]]) -> int:
        '''
        Updates the reward function using simulation data.
        collected_states is a list of (start_state, end_state, transition) tuples.
        controllers gives a list of controllers for each mode (supporting ensemble models).
        Returns env steps for simulation.
        '''
        pass

    def make_serializable(self):
        '''
        Makes the object serializable for parallel learning.
        '''
        self.mode = None
        self.automaton = None

    def recover_after_serialization(self, automaton: HybridAutomaton):
        '''
        Restores the object after call to make_serializable().
        '''
        self.mode = automaton.modes[self.mname]
        self.automaton = automaton


class SVMReward(RewardFunc):
    '''
    Reward function that has an SVM model for predicting good exit states.
    '''

    def __init__(self, mode, automaton, time_limits, discount=0.95,
                 penalty_factor=1., bonus=10.):
        super().__init__(mode, automaton)
        self.time_limits = time_limits
        self.discount = discount
        self.svm_model = None
        self.num_updates = 0
        self.penalty_factor = penalty_factor
        self.bonus = bonus

    def obs_reward(self, obs, action, next_obs, orig_reward, is_success, jump_obs=None):

        # compute classifier bonus
        if self.svm_model is not None and is_success:
            pred_y = self.svm_model.predict(np.array([self.mode.normalize_exit_state(next_obs)]))[0]
            orig_reward += ((self.bonus * pred_y) - (self.num_updates *
                                                     self.penalty_factor * (1 - pred_y)))

        return orig_reward

    def update(self, collected_states, controllers):
        state_value_success, steps_taken = self._evaluate_states(collected_states, controllers)

        if len(state_value_success) > 10 and \
                np.any([success for _, _, success in state_value_success]) and \
                np.any([not success for _, _, success in state_value_success]):

            print('Training SVM model...')
            start_time = time.time()

            # form training data
            X = np.array([self.mode.normalize_exit_state(self.mode.observe(s))
                          for s, _, _ in state_value_success])
            Y = np.array([label for _, _, label in state_value_success], dtype=np.int32)

            # train SVM model
            self.svm_model = svm.LinearSVC()
            self.svm_model.fit(X, Y)

            self.num_updates += 1

            print('Training SVM model completed in {} secs'.format(time.time() - start_time))

        return steps_taken

    def _evaluate_states(self, collected_states, controllers):
        state_value_success = []
        steps_taken = 0

        # compute values for all end states
        for start_state, end_state, transition in collected_states:

            if transition is not None:
                value = 1e9
                success = True

                # evaluate end state
                for target in transition.targets:

                    # Get next mode and starting state
                    next_mode = self.automaton.modes[target]
                    state = transition.jump(target, end_state)
                    target_success = False
                    target_reward = -1e9

                    for controller in controllers[target]:
                        # obtain rollout
                        sass, info = get_rollout(
                            next_mode, self.automaton.transitions[target], controller,
                            state, max_timesteps=self.time_limits[target])
                        steps_taken += len(sass)

                        # compute discounted reward
                        reward = 0
                        for sas in reversed(sass):
                            reward = self.__call__(*sas) + self.discount * reward
                        target_reward = max(target_reward, reward)

                        # evaluate success
                        target_success = target_success or (
                            info['safe'] and (info['jump'] is not None))

                    value = min(value, target_reward)
                    success = success and target_success

                state_value_success.append((end_state, value, success))

        return state_value_success, steps_taken


class ValueBasedReward(RewardFunc):

    def __init__(self, mode, automaton, discount=0.99, bonus=25.,
                 deterministic=False):
        super().__init__(mode, automaton)
        self.discount = discount
        self.bonus = bonus
        self.value_fns = None
        self.deterministic = deterministic

    def obs_reward(self, obs, action, next_obs, orig_reward, is_success, jump_obs=None):

        # compute classifier bonus
        if self.value_fns is not None and is_success:
            orig_reward = orig_reward + self.bonus + \
                self.discount * self.compute_value(jump_obs)

        return orig_reward

    def update(self, collected_states, controllers):
        value_fns = {}
        for mname, c_list in controllers.items():
            value_fns[mname] = [c.copy() for c in c_list]
        self.value_fns = value_fns
        return 0

    def compute_value(self, jump_obs):
        val = np.inf
        for mname, obs in jump_obs:
            j_val = -np.inf
            for value_fn in self.value_fns[mname]:
                j_val = max(j_val, value_fn.get_value(obs, self.deterministic))
            val = min(val, j_val)
        return val
