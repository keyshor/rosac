'''
Falsification using RL
'''

import gym
import time
import random
import numpy as np

from typing import Dict, Union, Tuple, List, Any, Optional
from hybrid_gym.hybrid_env import HybridAutomaton
from hybrid_gym.model import Controller, Mode, ModeSelector, Transition
from hybrid_gym.selectors import MaxJumpWrapper
from hybrid_gym.util.test import get_rollout
from stable_baselines.deepq import DQN
from stable_baselines.common.base_class import BaseRLModel


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
                 observation_type: str = 'full') -> None:
        self.automaton = automaton
        self.controller = controller
        self.reset_every_step = not isinstance(self.controller, Controller)
        self.obs_type = observation_type
        self.time_limits = time_limits
        self.max_timesteps = max_timesteps
        self.int_to_mode, self.mode_to_int = self._mode_mappings()

        self.action_space = gym.spaces.Discrete(len(self.int_to_mode))
        self.observation_space = self._get_obs_space(self.obs_type)

        mode_list = [mode.name for mode in self.int_to_mode]
        self.init_transition = InitTransition('', self.automaton.modes, mode_list)

    def reset(self):
        self.t = 0
        self.mode = random.choice(self.int_to_mode)
        self.mname = self.mode.name
        self.state = self.mode.reset()
        self.transition = self.init_transition
        return self.get_obs()

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

        return self.get_obs(), reward, done, {}

    def render(self) -> None:
        print(self.get_obs())

    def get_obs(self, obs_type: Optional[str] = None, mname: Optional[str] = None,
                state: Any = None):

        # Handle missing inputs
        if obs_type is None:
            obs_type = self.obs_type
        if mname is None:
            mname = self.mname
        if state is None:
            state = self.state

        if obs_type == 'mode':
            return self.mode_to_int[mname]
        elif obs_type == 'obs':
            return self.automaton.modes[mname].observe(state)
        elif obs_type == 'state':
            return self.automaton.modes[mname].vectorize_state(state)
        elif obs_type == 'mode_obs':
            return np.concatenate([self._one_hot(mname), self.get_obs('obs', mname, state)])
        elif obs_type == 'full':
            return np.concatenate([self._one_hot(mname), self.get_obs('state', mname, state)])

    def _one_hot(self, mname: str) -> np.ndarray:
        one_hot = np.zeros((len(self.int_to_mode),))
        one_hot[self.get_obs('mode', mname)] = 1.
        return one_hot

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
            space2 = self._get_obs_space('obs')
            shape = (len(self.int_to_mode) + space2.shape[0],)
            return gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape)
        elif obs_type == 'full':
            space2 = self._get_obs_space('state')
            shape = (len(self.int_to_mode) + space2.shape[0],)
            return gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape)


class InitTransition(Transition):
    '''
    Initial dummy transition that can transition to any mode.
    '''

    def __init__(self, source: str, modes: Dict[str, Mode], mode_list: List[str]) -> None:
        self.modes = modes
        super().__init__(source, mode_list)

    def guard(self, state: Any) -> bool:
        return True

    def jump(self, target: str, state: Any) -> Any:
        return self.modes[target].end_to_end_reset()


class RLSelector(ModeSelector):
    '''
    Mode Selector corresponding to a Stable Baselines model.
    '''

    def __init__(self, model: BaseRLModel, env: SelectorEnv):
        self.model = model
        self.env = env

    def next_mode(self, transition: Transition, state: Any) -> Tuple[str, bool]:
        action, _ = self.model.predict(self.env.get_obs(mname=transition.source, state=state))
        mode_idx = action % len(transition.targets)
        return transition.targets[mode_idx], False

    def reset(self) -> str:
        mode_idx, _ = self.model.predict(self.env.reset())
        return self.env.init_transition.targets[mode_idx]


class MCTS_Node:

    def __init__(self, num_branches, parent=None):
        self.num_branches = num_branches
        self.parent = parent
        self.children = None
        self.q_val = 0.
        self.num_visits = 0

    def get_ucb_action(self, exploration_constant):
        if self.children is not None:
            max_children = []
            max_val = -np.inf
            for c in range(self.num_branches):
                ucb_val = self.children[c]._get_ucb_val(exploration_constant)
                if ucb_val > max_val:
                    max_val = ucb_val
                    max_children = [c]
                elif ucb_val == max_val:
                    max_children.append(c)
            return random.choice(max_children)
        else:
            return random.choice(range(self.num_branches))

    def get_child(self, action):
        if self.num_visits == 0:
            return None
        if self.children is None:
            self.create_children()
        return self.children[action]

    def backpropagate(self, reward):
        self.q_val += reward
        self.num_visits += 1
        if self.parent is not None:
            self.parent.backpropagate(reward)

    def get_best_child(self):
        if self.children is not None:
            qvals = [child.q_val for child in self.children]
            return np.argmax(qvals)
        else:
            return -1

    def create_children(self):
        self.children = [MCTS_Node(self.num_branches, parent=self)
                         for _ in range(self.num_branches)]

    def _get_ucb_val(self, C):
        try:
            exploit_val = (self.q_val / self.num_visits)
            explore_val = C * np.sqrt(np.log(self.parent.num_visits) / self.num_visits)
            return exploit_val + explore_val
        except ZeroDivisionError:
            return np.inf


class MCTS_Selector(ModeSelector):
    '''
    Mode Selector corresponding to a MCTS tree.
    '''
    root: MCTS_Node
    node: MCTS_Node
    env: SelectorEnv
    mname: str

    def __init__(self, root: MCTS_Node, env: SelectorEnv, max_jumps: int):
        self.root = root
        self.env = env
        self.max_jumps = max_jumps

    def next_mode(self, transition: Transition, state: Any) -> Tuple[str, bool]:
        if self.node.children is None:
            self.mname = random.choice(transition.targets)
        else:
            mode_idx = self.node.get_best_child()
            self.node = self.node.children[mode_idx]
            mode_idx = mode_idx % len(transition.targets)
            self.mname = transition.targets[mode_idx]
        self.jumps += 1
        return self.mname, self.jumps >= self.max_jumps

    def reset(self) -> str:
        mode_idx = self.root.get_best_child()
        self.mname = self.env.init_transition.targets[mode_idx]
        self.node = self.root.children[mode_idx]
        self.jumps = 0
        return self.mname


def dqn_adversary(automaton: HybridAutomaton,
                  controller: Union[Dict[str, Controller], Controller],
                  time_limits: Dict[str, int],
                  max_jumps: int = 100,
                  observation_type: str = 'full',
                  learning_timesteps: int = 1000,
                  policy_kwargs: Optional[Dict] = None,
                  dqn_kwargs: Dict = {}) -> ModeSelector:
    '''
    Learns an adversary to pick the next mode during each mode transition.

    Inputs:
        time_limits: maximum steps allowed to complete each mode
        max_jumps: maximum number of mode transitions
        observation_type: either 'mode', 'obs', 'state', 'mode_obs' or 'full'
        kwargs: hyperparams for DQN or Q-learning
    '''
    adversary_env = SelectorEnv(automaton, controller, time_limits, max_timesteps=max_jumps,
                                observation_type=observation_type)
    model = DQN('MlpPolicy', adversary_env, policy_kwargs=policy_kwargs, **dqn_kwargs)
    model.learn(learning_timesteps)
    return MaxJumpWrapper(RLSelector(model, adversary_env), max_jumps)


def mcts_adversary(automaton: HybridAutomaton,
                   controller: Union[Dict[str, Controller], Controller],
                   time_limits: Dict[str, int],
                   max_jumps: int = 100,
                   exploration_constant: float = 1.414,
                   num_rollouts: int = 1000,
                   print_debug=False):

    # Create adversary env
    env = SelectorEnv(automaton, controller, time_limits, max_timesteps=max_jumps)

    # Create empty tree
    num_branches = env.action_space.n
    root = MCTS_Node(num_branches)
    root.create_children()

    # Perform rollouts and extend tree
    print('Performing MCTS...')
    start_time = time.time()
    for _ in range(num_rollouts):

        reward = 0.
        done = False
        node = root
        leaf = node
        env.reset()
        path = 'root'

        while not done:

            # select action and update node
            if node is not None:
                action = node.get_ucb_action(exploration_constant)
                node = node.get_child(action)
                if node is not None:
                    leaf = node
            else:
                action = random.choice(range(num_branches))

            # Step environment
            _, r, done, _ = env.step(action)
            path += ' -> {}'.format(env.mname)

            # update reward
            reward += float(r > 0)

        if print_debug:
            print(path)
        leaf.backpropagate(reward)

    print('Completed MCTS in {} secs'.format(time.time() - start_time))
    # return mcts based selector
    return MCTS_Selector(root, env, max_jumps)
