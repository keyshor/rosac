import numpy as np
import gym
import math

from hybrid_gym.model import Mode
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
        self.vdoor = np.array(vertical_door) + self.wall_size[1]
        self.hdoor = np.array(horizontal_door) + self.wall_size[0]

    def sample_full(self):
        return (np.random.random_sample(2) * self.room_size) - (self.room_size / 2)

    def sample_center(self):
        return (np.random.random_sample(2) * self.wall_size) - (self.wall_size / 2)


class RoomsMode(Mode[Tuple[Tuple, Tuple]]):

    def __init__(self, grid_params: GridParams, name: str):
        self.grid_params = grid_params
        self._goal = self._get_goal(name)

        # Compute scaling of action for normalization
        max_vel = np.amin(self.grid_params.wall_size) / 2
        self.action_scale = np.array([max_vel, np.pi/2])

        # Define action space
        high = np.array([1., 1.])
        low = -high
        action_space = gym.spaces.Box(low, high, dtype=np.float32)

        # Define observation space
        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,))

        # Initialize super
        super().__init__(name, action_space, observation_space)

    def reset(self):
        pos = tuple(self.grid_params.sample_full())
        return (pos, pos)

    def end_to_end_reset(self):
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
            / np.mean(self.grid_params.partition_size)
        safety_reward = 0.
        if not self.is_safe(next_state):
            safety_reward = -50.
        return reach_reward + safety_reward

    def vectorize_state(self, state):
        return np.concatenate(list(state))

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

    # check if line from s1 to s2 intersects the horizontal axis at a point inside door region
    # horizontal coordinates should be relative positions within rooms
    def check_horizontal_intersect(self, s1, s2, x):
        y = ((s2[1] - s1[1]) * (x - s1[0]) / (s2[0] - s1[0])) + s1[1]
        return (self.grid_params.vdoor[0] <= y and y <= self.grid_params.vdoor[1])

    # check if line from s1 to s2 intersects the vertical axis at a point inside door region
    # vertical coordinates should be relative positions within rooms
    def check_vertical_intersect(self, s1, s2, y):
        x = ((s2[0] - s1[0]) * (y - s1[1]) / (s2[1] - s1[1])) + s1[0]
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
        return s - self._get_goal(direction)

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
        return np.array(goal) * self.grid_params.partition_size
