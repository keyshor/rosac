import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from gym import spaces
from hybrid_gym.model import Mode
import enum
from typing import NamedTuple, List, Dict, Tuple

CRASH_REWARD: float = -10
PROGRESS_GAIN: float = 10

FloatPair = Tuple[float, float]
FloatQuad = Tuple[float, float, float, float]

class State(NamedTuple):
    x: float
    y: float

class Direction(enum.Enum):
    UP = enum.auto()
    RIGHT = enum.auto()
    DOWN = enum.auto()
    LEFT = enum.auto()

def make_goal(d: Direction, r: float) -> FloatPair:
    return (0, 1.1*r) if d == Direction.UP \
    else (1.1*r, 0) if d == Direction.RIGHT \
    else (0, -1.1*r) if d == Direction.DOWN \
    else (-1.1*r, 0) # if d == Direction.LEFT

def make_goal_region(d: Direction, rr: float, dr: float) -> FloatQuad:
    return (-dr, dr, rr, 2*rr) if d == Direction.UP \
    else (rr, 2*rr, -dr, dr) if d == Direction.RIGHT \
    else (-dr, dr, -2*rr, -rr) if d == Direction.DOWN \
    else (-2*rr, -rr, -dr, dr) # if d == Direction.LEFT

def make_start_box(d: Direction, rr: float, dr: float) -> FloatQuad:
    return (-dr, dr, 0.8*rr, rr) if d == Direction.UP \
    else (0.8*rr, rr, -dr, dr) if d == Direction.RIGHT \
    else (-dr, dr, -rr, -0.8*rr) if d == Direction.DOWN \
    else (-rr, -0.8*rr, -dr, dr) # if d == Direction.LEFT

class RoomsMode(Mode[State]):
    room_radius: float
    door_radius: float
    action_scale: float
    direction: Direction
    goal_x: float
    goal_y: float
    start_boxes: List[FloatQuad]
    goal_region: FloatQuad
    rng: np.random.Generator

    def __init__(self,
                 name: str,
                 direction: Direction,
                 room_radius: float,
                 door_radius: float,
                 action_scale: float,
                 rng: np.random.Generator = np.random.default_rng(),
                 ) -> None:
        assert door_radius <= room_radius, \
            f'door radius {door_radius} is larger than room radius {room_radius}'
        self.room_radius = room_radius
        self.door_radius = door_radius
        self.direction = direction
        self.goal_x, self.goal_y = make_goal(direction, room_radius)
        self.start_boxes = [
            make_start_box(d, room_radius, door_radius)
            for d in Direction if d != direction
        ]
        self.goal_region = make_goal_region(direction, room_radius, door_radius)
        self.action_scale = action_scale
        self.rng = rng
        super().__init__(
            name=name,
            action_space=spaces.Box(low=-1.0, high=1.0,
                                    shape=(2,), dtype=np.float32),
            observation_space=spaces.Box(low=-2*room_radius, high=2*room_radius,
                                         shape=(4,), dtype=np.float32),
        )

    def reset(self) -> State:
        x_low, x_high, y_low, y_high = self.rng.choice(self.start_boxes)
        return State(
            x = self.rng.uniform(x_low, x_high),
            y = self.rng.uniform(y_low, y_high),
        )

    def is_safe(self, st: State) -> bool:
        goal_x_low, goal_x_high, goal_y_low, goal_y_high = self.goal_region
        return np.all(np.array([np.abs(st.x), np.abs(st.y)]) <= self.room_radius) \
            or (goal_x_low <= st.x <= goal_x_high and goal_y_low <= st.y <= goal_y_high)

    def render(self, st: State) -> None:
        fig, ax = plt.subplots()
        self.plot_walls(ax)
        ax.plot([self.goal_x], [self.goal_y], color='g', marker='o')
        ax.plot([st.x], [st.y], color='r', marker='x')
        plt.show()

    def _step_fn(self, st: State, action: np.ndarray) -> State:
        scaled_action = self.action_scale * action
        return State(
            x = st.x + scaled_action[0],
            y = st.y + scaled_action[1],
        )

    def _observation_fn(self, st: State) -> np.ndarray:
        return np.array([st.x, st.y, self.goal_x, self.goal_y])

    def _reward_fn(self, st0: State, action: np.ndarray, st1: State) -> float:
        if not self.is_safe(st1):
            return CRASH_REWARD
        return PROGRESS_GAIN * (self.goal_dist(st0) - self.goal_dist(st1))

    def vectorize_state(self, st: State) -> np.ndarray:
        return np.array([st.x, st.y])

    def state_from_vector(self, vec: np.ndarray) -> State:
        return State(x=vec[0], y=vec[1])

    def goal_dist(self, st: State) -> float:
        return np.abs(st.x - self.goal_x) + np.abs(st.y - self.goal_y)

    def plot_walls(self, ax: mpl.axes.Axes) -> None:
        # top left corner
        ax.add_line(mpl.lines.Line2D(
            xdata=[-self.room_radius, -self.room_radius, -self.door_radius],
            ydata=[self.door_radius, self.room_radius, self.room_radius],
        ))
        # top right corner
        ax.add_line(mpl.lines.Line2D(
            xdata=[self.door_radius, self.room_radius, self.room_radius],
            ydata=[self.room_radius, self.room_radius, self.door_radius],
        ))
        # bottom left corner
        ax.add_line(mpl.lines.Line2D(
            xdata=[-self.room_radius, -self.room_radius, -self.door_radius],
            ydata=[-self.door_radius, -self.room_radius, -self.room_radius],
        ))
        # bottom right corner
        ax.add_line(mpl.lines.Line2D(
            xdata=[self.door_radius, self.room_radius, self.room_radius],
            ydata=[-self.room_radius, -self.room_radius, -self.door_radius],
        ))
