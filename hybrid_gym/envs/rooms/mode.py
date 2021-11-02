import numpy as np
import gym
import math

from hybrid_gym.model import Mode
from hybrid_gym.synthesis.abstractions import Box, StateWrapper
from typing import Tuple


class GridParams:
    '''
    Parameters for defining the rooms environment.
    '''

    def __init__(self, room_size, wall_size, horizontal_door, vertical_door):
        '''
        room_size: (b:int, l:int) size of a single room
        wall_size: (tx:int, ty:int) thickness of walls (thickness of vertical wall first)
        vertical_door, horizontal_door: relative coordinates for door, specifies min and max
                                        coordinates for door space
        '''
        self.room_size = np.array(room_size)
        self.wall_size = np.array(wall_size)
        self.partition_size = self.room_size + self.wall_size
        self.full_size = self.partition_size + self.wall_size
        self.hdoor = np.array(horizontal_door) + self.wall_size[0]
        self.vdoor = np.array(vertical_door) + self.wall_size[1]
        self.center_size = np.array([self.hdoor[1] - self.hdoor[0], self.vdoor[1] - self.vdoor[0]])
        self.center_start = np.array([self.hdoor[0], self.vdoor[0]]) - (self.full_size/2)
        self.bd_size = np.array([self.center_size[0], self.wall_size[1]])
        self.bd_point = np.array([self.hdoor[0], 0]) - (self.full_size/2)
        self.full_init_size = np.array([self.center_size[0], self.vdoor[1]])

    def sample_full(self):
        if np.random.binomial(1, 0.75):
            return self.sample_bottom()
        else:
            return self.sample_center()

    def sample_center(self):
        return (np.random.random_sample(2) * self.center_size) + self.center_start

    def sample_bottom(self):
        return (np.random.random_sample(2) * self.bd_size) + self.bd_point

    def sample_bottom_center(self):
        return (np.random.random_sample(2) * (self.bd_size/2)) + (self.bd_point + (self.bd_size/4))


class RoomsMode(Mode[Tuple[Tuple, Tuple]]):

    def __init__(self, grid_params: GridParams, name: str, bottom_start: bool = False):
        self.grid_params = grid_params
        self.bottom_start = bottom_start
        self._goal = self._get_goal(name)
        self._reward_scale = np.mean(self.grid_params.partition_size)

        # Compute scaling of action for normalization
        max_vel = np.amin(self.grid_params.wall_size) / 2
        self.action_scale = np.array([max_vel, np.pi/2])

        # Define action space
        high = np.array([1., 1.])
        low = -high
        action_space = gym.spaces.Box(low, high)

        # Define observation space
        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,))

        # Initialize super
        super().__init__(name, action_space, observation_space)

    def reset(self):
        if self.bottom_start:
            pos = tuple(self.grid_params.sample_bottom())
        else:
            pos = tuple(self.grid_params.sample_full())
        return (pos, pos)

    def end_to_end_reset(self):
        if self.bottom_start:
            pos = tuple(self.grid_params.sample_bottom_center())
        else:
            pos = tuple(self.grid_params.sample_center())
        return (pos, pos)

    def render(self, state):
        print(state[1])

    def _step_fn(self, state, action):
        action = self.action_scale * action
        action = np.array([action[0] * math.cos(action[1]),
                           action[0] * math.sin(action[1])])
        next_state = tuple(np.array(state[1]) + action)
        return (state[1], next_state)

    def _observation_fn(self, state):
        return np.array(state[1])

    def _reward_fn(self, state, action, next_state):
        reach_reward = -np.linalg.norm(self._goal - np.array(next_state[1])) \
            / self._reward_scale
        safety_reward = 0.
        if not self.is_safe(next_state):
            safety_reward = -25.
        return reach_reward + safety_reward

    def vectorize_state(self, state):
        return np.concatenate(state)

    def state_from_vector(self, vec):
        return (tuple(vec[:2]), tuple(vec[2:]))

    def is_safe(self, state):
        s1, s2 = state
        s1 = np.array(s1)
        s2 = np.array(s2)
        params = self.grid_params

        # Change of coordiates
        p1 = s1 + (params.full_size / 2)
        p2 = s2 + (params.full_size / 2)

        if not self.is_state_legal(p2):
            return False

        if self.hit_obstacle(p1, p2):
            return False

        # both states are inside the same room (not in the door area)
        if np.all(p1 <= params.partition_size) and np.all(p1 >= params.wall_size) and \
                np.all(p2 <= params.partition_size) and np.all(p2 >= params.wall_size):
            return True
        # both states outside the interior room area
        if (np.any(p1 >= params.partition_size) or np.any(p1 <= params.wall_size)) \
                and (np.any(p2 >= params.partition_size) or np.any(p2 <= params.wall_size)):
            # p2 outside the room
            if np.any(p2 <= 0.) or np.any(p2 >= params.full_size):
                direction = self.compute_direction(s2)
                return self.is_safe((self.change_of_coordinates(s1, direction),
                                     self.change_of_coordinates(s2, direction)))
            # both states in door area
            else:
                return True

        # swap p1 and p2 if p1 is in the doorway
        if (np.any(p1 >= params.partition_size) or np.any(p1 <= params.wall_size)):
            p1, p2 = p2, p1

        # four cases to consider
        if p2[0] > params.partition_size[0]:
            return self.check_horizontal_intersect(p1, p2, params.partition_size[0])
        elif p2[0] < params.wall_size[0]:
            return self.check_horizontal_intersect(p2, p1, params.wall_size[0])
        elif p2[1] > params.partition_size[1]:
            return self.check_vertical_intersect(p1, p2, params.partition_size[1])
        else:
            return self.check_vertical_intersect(p2, p1, params.wall_size[1])

    def is_state_legal(self, p):
        params = self.grid_params

        # legal if outside the room
        if np.any(p <= 0.) or np.any(p >= params.full_size):
            return True
        # legal if completely inside the room
        if np.all(p <= params.partition_size) and np.all(p >= params.wall_size):
            return True

        # reject positions within walls
        if p[0] > params.partition_size[0] or p[0] < params.wall_size[0]:
            return (p[1] >= params.vdoor[0] and p[1] <= params.vdoor[1])
        elif p[1] > params.partition_size[1] or p[1] < params.wall_size[1]:
            return (p[0] >= params.hdoor[0] and p[0] <= params.hdoor[1])

    def hit_obstacle(self, p1, p2):
        if self.grid_params.hdoor[0] <= p1[0] and p1[0] <= self.grid_params.hdoor[1]:
            if p1[1] < self.grid_params.wall_size[1] and p2[1] >= self.grid_params.wall_size[1]:
                x = ((p2[0] - p1[0]) * (self.grid_params.wall_size[1] - p1[1]) / (p2[1] - p1[1])) \
                    + p1[0]
                return x < self.grid_params.full_size[0]/2
        return False

    # check if line from s1 to s2 intersects the horizontal axis at a point inside door region
    # horizontal coordinates should be relative positions within rooms
    def check_horizontal_intersect(self, p1, p2, x):
        y = ((p2[1] - p1[1]) * (x - p1[0]) / (p2[0] - p1[0])) + p1[1]
        return (self.grid_params.vdoor[0] <= y and y <= self.grid_params.vdoor[1])

    # check if line from s1 to s2 intersects the vertical axis at a point inside door region
    # vertical coordinates should be relative positions within rooms
    def check_vertical_intersect(self, p1, p2, y):
        x = ((p2[0] - p1[0]) * (y - p1[1]) / (p2[1] - p1[1])) + p1[0]
        return (self.grid_params.hdoor[0] <= x and x <= self.grid_params.hdoor[1])

    def compute_direction(self, s):
        p = s + (self.grid_params.full_size / 2)
        if p[0] <= 0.:
            return 'left'
        if p[0] >= self.grid_params.full_size[0]:
            return 'right'
        if p[1] <= 0.:
            return 'down'
        if p[1] >= self.grid_params.full_size[1]:
            return 'up'
        return None

    def change_of_coordinates(self, s, direction):
        return s - (2*self._get_goal(direction))

    def completed_task(self, s):
        p = s + (self.grid_params.full_size / 2)
        if p[0] < self.grid_params.wall_size[0]:
            return 'left'
        if p[0] > self.grid_params.partition_size[0]:
            return 'right'
        if p[1] < self.grid_params.wall_size[1]:
            return 'down'
        if p[1] > self.grid_params.partition_size[1]:
            return 'up'
        return None

    def mode_transition(self, s, direction):
        if direction == 'left':
            return self.mode_transition(np.array([s[1], -s[0]]), 'up')
        if direction == 'right':
            return self.mode_transition(np.array([-s[1], s[0]]), 'up')
        if direction == 'up':
            return s - np.array([0, self.grid_params.partition_size[1]])
        if direction == 'down':
            return s
        return None

    def get_init_pre(self):
        if self.bottom_start:
            low = self.grid_params.bd_point + (self.grid_params.bd_size/4)
            high = low + (self.grid_params.bd_size/2)
        else:
            low = self.grid_params.center_start
            high = low + self.grid_params.center_size
        low = np.concatenate([low, low])
        high = np.concatenate([high, high])
        return StateWrapper(self, Box(low=low, high=high))

    def _get_goal(self, name):
        if name == 'left':
            goal = [-1, 0]
        elif name == 'right':
            goal = [1, 0]
        elif name == 'up':
            goal = [0, 1]
        elif name == 'down':
            goal = [0, -1]
        else:
            raise ValueError('Invalid mode name/direction!')
        return np.array(goal) * (self.grid_params.partition_size/2)
