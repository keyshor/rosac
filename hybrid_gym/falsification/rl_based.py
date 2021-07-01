'''
Falsification using RL
'''

import gym
import random
import numpy as np

from typing import Dict, Union, Tuple, List
from hybrid_gym.hybrid_env import HybridAutomaton
from hybrid_gym.model import Controller, Mode
from hybrid_gym.util.test import get_rollout


class SelectorEnv(gym.Env):
    '''
    Creates an environment where actions are mode selections.

    Parameters:
        observation_type - either 'mode', 'obs', 'state', 'mode_obs' or 'full'

    Reset - Tries to complete one mode 'max_reset_tries' times,
        returning state from which transition is trigerred (on success)
        raises exception on failure.
    Actions - Discrete with n = maximum number of choices for next mode in any state.
        If number of current choices (m) are less, the remainder modulo m is used.
    Rewards - (-1) every step until crash or unable to reach target set in a mode.
        (+10) bonus for making the system fail.
    '''

    def __init__(self, automaton: HybridAutomaton,
                 controller: Union[Dict[str, Controller], Controller],
                 time_limits: Dict[str, int],
                 max_timesteps: int = 100,
                 observation_type: str = 'mode',
                 max_reset_tries: int = 100) -> None:
        self.automaton = automaton
        self.controller = controller
        self.reset_every_step = not isinstance(self.controller, Controller)
        self.obs_type = observation_type
        self.time_limits = time_limits
        self.max_timesteps = max_timesteps
        self.max_reset_tries = max_reset_tries
        self.int_to_mode, self.mode_to_int = self._mode_mappings()

        num_actions = self._compute_max_choices()
        self.action_space = gym.spaces.Discrete(num_actions)
        self.observation_space = self._get_obs_space(self.obs_type)

    def reset(self):
        self.t = 0

        for _ in range(self.max_reset_tries):
            self.mode = random.choice(self.int_to_mode)
            self.mname = self.mode.name

            # pick controller to use
            if isinstance(self.controller, Controller):
                self.controller.reset()
                cur_controller = self.controller
            else:
                cur_controller = self.controller[self.mname]

            # get rollout to reach decision point for next mode
            sass, info = get_rollout(self.mode, self.automaton.transitions[self.mname],
                                     cur_controller, max_timesteps=self.time_limits[self.mname],
                                     reset_controller=self.reset_every_step)

            # return the appropriate observation on success
            if info['safe'] and info['jump'] is not None:
                self.transition = info['jump']
                self.state = sass[-1][-1]
                return self._get_obs(self.obs_type)

        raise Exception('No successful execution found after {} tries!'
                        .format(self.max_reset_tries))

    def step(self, action):
        mode_choice = action % len(self.transition.targets)
        self.mname = self.transition.targets[mode_choice]
        self.mode = self.automaton.modes[self.mname]
        start_state = self.transition.jump(self.mname, self.state)

        # pick controller to use
        if isinstance(self.controller, Controller):
            cur_controller = self.controller
        else:
            cur_controller = self.controller[self.mname]

        # get rollout to reach decision point for next mode
        sass, info = get_rollout(self.mode, self.automaton.transitions[self.mname],
                                 cur_controller, max_timesteps=self.time_limits[self.mname],
                                 reset_controller=self.reset_every_step,
                                 state=start_state)

        self.t += 1
        done = self.t >= self.max_timesteps
        reward = -1.

        if (not info['safe']) or (info['jump'] is None):
            done = True
            reward = 10.
            self.state = sass[-1][0]
        else:
            self.transition = info['jump']
            self.state = sass[-1][-1]

        return self._get_obs(self.obs_type), reward, done, {}

    def render(self):
        print(self._get_obs(self.obs_type))

    def _get_obs(self, obs_type: str):
        if obs_type == 'mode':
            return self.mode_to_int[self.mname]
        elif obs_type == 'obs':
            return self.mode.observe(self.state)
        elif obs_type == 'state':
            return self.mode.vectorize_state(self.state)
        elif obs_type == 'mode_obs':
            return (self._get_obs('mode'), self._get_obs('obs'))
        elif obs_type == 'full':
            return (self._get_obs('mode'), self._get_obs('state'))

    def _compute_max_choices(self) -> int:
        max_choices = 0
        for _, t_list in self.automaton.transitions.items():
            for transition in t_list:
                max_choices = max(max_choices, len(transition.targets))
        return max_choices

    def _mode_mappings(self) -> Tuple[List[Mode], Dict[str, int]]:
        int_to_mode: List[Mode] = []
        mode_to_int: Dict[str, int] = {}
        for mname, mode in self.automaton.modes.items():
            mode_to_int[mname] = len(int_to_mode)
            int_to_mode.append(mode)
        return int_to_mode, mode_to_int

    def _get_obs_space(self, obs_type: str) -> gym.spaces.Space:
        if obs_type == 'mode':
            return gym.spaces.Discrete(len(self.int_to_mode))
        elif obs_type == 'obs':
            return self.int_to_mode[0].observation_space
        elif obs_type == 'state':  # requires vectorize_state() to be implemented in all modes
            dummy_mode = self.int_to_mode[0]
            shape = dummy_mode.vectorize_state(dummy_mode.reset()).shape
            return gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape)
        elif obs_type == 'mode_obs':
            space1 = self._get_obs_space('mode')
            space2 = self._get_obs_space('obs')
            return gym.spaces.Tuple((space1, space2))
        elif obs_type == 'full':
            space1 = self._get_obs_space('mode')
            space2 = self._get_obs_space('state')
            return gym.spaces.Tuple((space1, space2))
