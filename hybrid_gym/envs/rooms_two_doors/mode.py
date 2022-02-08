import numpy as np
import gym
import math

from hybrid_gym.model import Mode
from hybrid_gym.synthesis.abstractions import Box, StateWrapper
from matplotlib import pyplot as plt
from typing import Tuple


class GridParams:
    '''
    Parameters for defining the rooms environment.
    '''

    def __init__(self, room_size, wall_size, horizontal_doors, vertical_door):
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
        self.hdoors = [np.array(horizontal_door) + self.wall_size[0]
                       for horizontal_door in horizontal_doors]
        self.vdoor = np.array(vertical_door) + self.wall_size[1]

        # Following coordinates are wrt center of room as origin
        self.bd_size = np.array([self.hdoors[1][1] - self.hdoors[1][0], self.wall_size[1]/2])
        self.bd_point = np.array([(self.full_size[0] - self.bd_size[0]) / 2,
                                  self.wall_size[1]/2]) - (self.full_size/2)
        self.bhdoor = np.array([self.bd_point[0], self.bd_point[0] + self.bd_size[0]]) \
            + self.full_size[0]/2
        self.exit_wall_size = 0.
        self.exit_opening_size = self.bd_size[0] - self.exit_wall_size
        self.exit_wall = self.bd_point[0] + (self.full_size[0]/2) + self.exit_wall_size

    def sample_bottom(self):
        return (np.random.random_sample(2) * self.bd_size) + self.bd_point

    def sample_bottom_center(self):
        return (np.random.random_sample(2) * (self.bd_size/2)) + (self.bd_point + (self.bd_size/4))

    def sample_bottom_strip(self):
        return np.array([np.random.random_sample(1)[0] * (self.exit_opening_size) + self.bd_point[0]
                         + self.exit_wall_size, self.bd_point[1] + self.wall_size[1]/2])

    def plot_vertical_walls(self, N):
        # TODO: Modify this for two doors
        const_arr = np.ones((N,))

        # walls
        outer_walls = []
        inner_walls = []
        door_walls = []

        # outer walls
        outer_walls.append(np.linspace(0, self.vdoor[0], N))
        outer_walls.append(np.linspace(self.vdoor[1], self.full_size[1], N))

        # inner walls
        inner_walls.append(np.linspace(self.wall_size[1], self.vdoor[0], N))
        inner_walls.append(np.linspace(self.vdoor[1], self.partition_size[1], N))

        # door walls
        door_walls.append(np.linspace(0, self.wall_size[1], N))
        door_walls.append(np.linspace(self.partition_size[1], self.full_size[1], N))

        for w in outer_walls:
            plt.plot(0*const_arr, w, color='black')
            plt.plot(self.full_size[0]*const_arr, w, color='black')

        for w in inner_walls:
            plt.plot(self.wall_size[0]*const_arr, w, color='black')
            plt.plot(self.partition_size[0]*const_arr, w, color='black')

        for w in door_walls:
            plt.plot(self.hdoor[0]*const_arr, w, color='black')
            plt.plot(self.hdoor[1]*const_arr, w, color='black')

    def plot_horizontal_walls(self, N):
        # TODO: Modify this for two doors
        const_arr = np.ones((N,))

        # walls
        outer_walls = []
        inner_walls = []
        door_walls = []

        # outer walls
        outer_walls.append(np.linspace(0, self.hdoor[0], N))
        outer_walls.append(np.linspace(self.hdoor[1], self.full_size[0], N))

        # inner walls
        inner_walls.append(np.linspace(self.wall_size[0], self.hdoor[0], N))
        inner_walls.append(np.linspace(self.hdoor[1], self.partition_size[0], N))

        # door walls
        door_walls.append(np.linspace(0, self.wall_size[0], N))
        door_walls.append(np.linspace(self.partition_size[0], self.full_size[0], N))

        for w in outer_walls:
            plt.plot(w, 0*const_arr, color='black')
            plt.plot(w, self.full_size[1]*const_arr, color='black')

        for w in inner_walls:
            plt.plot(w, self.wall_size[1]*const_arr, color='black')
            plt.plot(w, self.partition_size[1]*const_arr, color='black')

        for w in door_walls:
            plt.plot(w, self.vdoor[0]*const_arr, color='black')
            plt.plot(w, self.vdoor[1]*const_arr, color='black')

    def plot_room(self):
        N = 10000
        ax = plt.gca()
        ax.set_aspect(1)
        self.plot_vertical_walls(N)
        self.plot_horizontal_walls(N)

        # exit obstacle
        obstacle = np.linspace(self.hdoor[0], self.exit_wall, N)
        plt.plot(obstacle, np.ones((N,))*self.wall_size[1], color='red')


class RoomsMode(Mode[Tuple[Tuple, Tuple]]):

    def __init__(self, grid_params: GridParams, name: str):
        self.grid_params = grid_params
        self._goals = self._get_goals(name)
        self._reward_scale = np.mean(self.grid_params.partition_size)

        # Compute scaling of action for normalization
        max_vel1 = np.amin(self.grid_params.wall_size) / 2
        max_vel2 = np.amin([hdoor[1] - hdoor[0] for hdoor in self.grid_params.hdoors]) / 2
        max_vel = min(max_vel1, max_vel2,
                      (self.grid_params.vdoor[1] - self.grid_params.vdoor[0]) / 2)
        self.action_scale = np.array([max_vel, np.pi/2])

        # Define action space
        high = np.array([1., 1.])
        low = -high
        action_space = gym.spaces.Box(low, high)

        # Define observation space
        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,))

        # Initialize super
        super().__init__(name, action_space, observation_space)
        self._set_exit_norm_factors()

    def reset(self):
        pos = tuple(self.grid_params.sample_bottom())
        return (pos, pos)

    def end_to_end_reset(self):
        pos = tuple(self.grid_params.sample_bottom_strip())
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
        return 2 * np.array(state[1]) / self.grid_params.full_size

    def _reward_fn(self, state, action, next_state):
        return np.amax([self.reward_fn_goal(state, action, next_state, goal)
                        for goal in self._goals])

    def reward_fn_goal(self, state, action, next_state, goal):
        reach_reward = -np.linalg.norm(goal - np.array(next_state[1])) \
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
            return self.check_vertical_intersect_top(p1, p2, params.partition_size[1])
        else:
            return self.check_vertical_intersect_bottom(p2, p1, params.wall_size[1])

    def is_state_legal(self, p):
        params = self.grid_params

        # legal if outside the room
        if np.any(p < 0.) or np.any(p > params.full_size):
            return False
        # legal if completely inside the room
        if np.all(p <= params.partition_size) and np.all(p >= params.wall_size):
            return True

        # reject positions within walls
        if p[0] > params.partition_size[0] or p[0] < params.wall_size[0]:
            return (p[1] >= params.vdoor[0] and p[1] <= params.vdoor[1])
        elif p[1] < params.wall_size[1]:
            return (p[0] >= params.bhdoor[0] and p[0] <= params.bhdoor[1])
        elif p[1] > params.partition_size[1]:
            return ((p[0] >= params.hdoors[0][0] and p[0] <= params.hdoors[0][1]) or
                    (p[0] >= params.hdoors[1][0] and p[0] <= params.hdoors[1][1]))

    def hit_obstacle(self, p1, p2):
        if self.grid_params.bhdoor[0] <= p1[0] and p1[0] <= self.grid_params.bhdoor[1]:
            if p1[1] < self.grid_params.wall_size[1] and p2[1] >= self.grid_params.wall_size[1]:
                x = ((p2[0] - p1[0]) * (self.grid_params.wall_size[1] - p1[1]) / (p2[1] - p1[1])) \
                    + p1[0]
                return x < self.grid_params.exit_wall
        return False

    # check if line from s1 to s2 intersects the horizontal axis at a point inside door region
    # horizontal coordinates should be relative positions within rooms
    def check_horizontal_intersect(self, p1, p2, x):
        y = ((p2[1] - p1[1]) * (x - p1[0]) / (p2[0] - p1[0])) + p1[1]
        return (self.grid_params.vdoor[0] <= y and y <= self.grid_params.vdoor[1])

    # check if line from s1 to s2 intersects the vertical axis at a point inside door region
    # vertical coordinates should be relative positions within rooms
    def check_vertical_intersect_bottom(self, p1, p2, y):
        x = ((p2[0] - p1[0]) * (y - p1[1]) / (p2[1] - p1[1])) + p1[0]
        return (self.grid_params.bhdoor[0] <= x and x <= self.grid_params.bhdoor[1])

    # check if line from s1 to s2 intersects the vertical axis at a point inside door region
    # vertical coordinates should be relative positions within rooms
    def check_vertical_intersect_top(self, p1, p2, y):
        x = ((p2[0] - p1[0]) * (y - p1[1]) / (p2[1] - p1[1])) + p1[0]
        return np.any([hdoor[0] <= x and x <= hdoor[1] for hdoor in self.grid_params.hdoors])

    def completed_task(self, s):
        p = s + (self.grid_params.full_size / 2)
        if p[0] < self.grid_params.wall_size[0] / 2:
            return 'left'
        if p[0] > self.grid_params.partition_size[0] + self.grid_params.wall_size[0] / 2:
            return 'right'
        if p[1] > self.grid_params.partition_size[1] + self.grid_params.wall_size[1] / 2:
            return 'up'
        return None

    def mode_transition(self, s, direction):
        if direction == 'left':
            return np.array([s[1], -s[0]]) - np.array([0, self.grid_params.partition_size[1]])
        if direction == 'right':
            return np.array([-s[1], s[0]]) - np.array([0, self.grid_params.partition_size[1]])
        if direction == 'up':
            if s[0] < (self.grid_params.hdoors[1][0] - self.grid_params.full_size[0]/2):
                return -self.grid_params.full_size/2
            return s - np.array([
                np.mean(self.grid_params.hdoors[1]) - (self.grid_params.full_size[0]/2),
                self.grid_params.partition_size[1]])
        return None

    def get_init_pre(self):
        low = self.grid_params.bd_point + \
            np.array([self.grid_params.exit_wall_size, self.grid_params.wall_size[1]/2])
        high = low + np.array([self.grid_params.exit_opening_size, 0])
        low = np.concatenate([low, low])
        high = np.concatenate([high, high])
        return StateWrapper(self, Box(low=low, high=high))

    def _set_exit_norm_factors(self):
        mean = np.zeros((2,))
        std = np.ones((2,))
        if self.name == 'left':
            mean[0] = -self.grid_params.partition_size[0]/2
            std[0] = self.grid_params.wall_size[0]/2
            std[1] = (self.grid_params.vdoor[1] - self.grid_params.vdoor[0])/2
        elif self.name == 'right':
            mean[0] = self.grid_params.partition_size[0]/2
            std[0] = self.grid_params.wall_size[0]/2
            std[1] = (self.grid_params.vdoor[1] - self.grid_params.vdoor[0])/2
        elif self.name == 'up':
            mean[1] = self.grid_params.partition_size[1]/2
            std[0] = (self.grid_params.hdoors[1][1] - self.grid_params.hdoors[0][0])/2
            std[1] = self.grid_params.wall_size[1]/2
        self.exit_mean = mean
        self.exit_std = std

    def normalize_exit_state(self, obs: np.ndarray) -> np.ndarray:
        return (np.array(obs) - self.exit_mean) / self.exit_std

    def _get_goals(self, name):
        goal_dist = (self.grid_params.partition_size / 2)
        if name == 'left':
            goals = [[-goal_dist, 0]]
        elif name == 'right':
            goals = [[goal_dist, 0]]
        elif name == 'up':
            door_centers = [((hdoor[1] + hdoor[0]) / 2) - (self.grid_params.full_size[0] / 2)
                            for hdoor in self.grid_params.hdoors]

            goals = [[dc, goal_dist] for dc in door_centers]
        else:
            raise ValueError('Invalid mode name/direction!')
        return np.array(goals)
