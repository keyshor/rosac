import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from gym import spaces
from hybrid_gym.model import Mode
from typing import List, Iterable, Tuple, Optional, NamedTuple, Type, TypeVar

FloatPair = Tuple[float, float]

CAR_LENGTH: float = .45  # in m
CAR_CENTER_OF_MASS: float = .225  # from rear of car (m)
CAR_DECEL_CONST: float = .4
CAR_ACCEL_CONST: float = 1.633  # estimated from data
CAR_MOTOR_CONST: float = 0.2  # estimated from data
HYSTERESIS_CONSTANT: float = 4
MAX_TURNING_INPUT: float = float(np.radians(20))  # in radians
CONST_THROTTLE: float = 16
MAX_THROTTLE: float = 30

LIDAR_RANGE: float = 5  # in m
SAFE_DISTANCE: float = 0.1  # in m
LIDAR_FIELD_OF_VIEW: float = float(np.radians(115))
TIME_STEP: float = 0.1
TIME_CHECK: np.ndarray = np.linspace(0, TIME_STEP, 100)
EKAT: np.ndarray = np.array(np.exp(-CAR_ACCEL_CONST * TIME_CHECK))

STEP_REWARD_GAIN: float = 5
CRASH_REWARD: float = -100
MINIMUM_ACCEPTABLE_SPEED: float = 1.5
LOW_SPEED_REWARD: float = -20
SPEED_GAIN: float = 0.0
PROGRESS_GAIN: float = 10
MIDDLE_GAIN: float = -10
DEFAULT_HALL_WIDTH: float = 1.5

EPSILON: float = 1e-10

def sgnstar(arr: np.ndarray) -> np.ndarray:
    return np.where(np.signbit(arr), -1, 1)

def first_true(arr: np.ndarray) -> int:
    # takes a 1-D boolean array and returns the position of the first True
    # returns -1 if all elements of arr are False
    try:
        return np.nonzero(arr)[0][0]
    except IndexError:
        return -1

def add_to_tuples(it_it_fp: Iterable[Iterable[FloatPair]],
                  dx: float, dy: float,
                  ) -> List[List[FloatPair]]:
    return [[(px + dx, py + dy) for (px, py) in it_fp]
            for it_fp in it_it_fp]

LineSegmentsTypevar = TypeVar('LineSegmentsTypevar', bound='LineSegments')

class LineSegments:
    seg_x1: np.ndarray
    seg_y1: np.ndarray
    seg_x2: np.ndarray
    seg_y2: np.ndarray
    # coefficients a, b, c
    # such that a*x + b*y + c = 0 and a^2 + b^2 = 1
    # for the line passing through each segment
    line_a: np.ndarray
    line_b: np.ndarray
    line_c: np.ndarray
    # diff_x = x2 - x1
    # diff_y = y2 - y1
    diff_x: np.ndarray
    diff_y: np.ndarray

    def __init__(self,
                 seg_x1: np.ndarray, seg_y1: np.ndarray,
                 seg_x2: np.ndarray, seg_y2: np.ndarray,
                 line_a: np.ndarray, line_b: np.ndarray, line_c: np.ndarray,
                 diff_x: np.ndarray, diff_y: np.ndarray,
                 ) -> None:
        self.seg_x1 = seg_x1
        self.seg_y1 = seg_y1
        self.seg_x2 = seg_x2
        self.seg_y2 = seg_y2
        self.line_a = line_a
        self.line_b = line_b
        self.line_c = line_c
        self.diff_x = diff_x
        self.diff_y = diff_y

    @classmethod
    def from_polygons(cls: Type[LineSegmentsTypevar],
                      polygons: Iterable[Iterable[FloatPair]],
                      paths: Iterable[Iterable[FloatPair]],
                      ) -> LineSegmentsTypevar:
        # if an iterable [p1, p2, ..., pn] appears in polygons,
        # then we have segments (p1, p2), (p2, p3), ..., (p{n-1}, pn), (pn, p1)
        #
        # if an iterable [p1, p2, ..., pn] appears in paths,
        # then we have segments (p1, p2), (p2, p3), ..., (p{n-1}, pn)
        segment_list: List[Tuple[FloatPair, FloatPair]] = []
        for polygon in polygons:
            polygon_list = list(polygon)
            shifted_list = polygon_list[1:] + [polygon_list[0]]
            segment_list += list(zip(polygon_list, shifted_list))
        for path in paths:
            path_list = list(path)
            segment_list += list(zip(path_list[:-1], path_list[1:]))
        if not segment_list:
            return cls.empty()
        segment_array = np.array([
            [x1, y1, x2, y2] for ((x1, y1), (x2, y2)) in segment_list
        ])

        seg_x1 = segment_array[:,0]
        seg_y1 = segment_array[:,1]
        seg_x2 = segment_array[:,2]
        seg_y2 = segment_array[:,3]

        line_a_unnormalized = seg_y1 - seg_y2
        line_b_unnormalized = seg_x2 - seg_x1
        line_c_unnormalized = seg_x1 * seg_y2 - seg_x2 * seg_y1
        line_norm = np.sqrt(np.square(line_a_unnormalized) + np.square(line_b_unnormalized))

        return cls(
            seg_x1 = seg_x1,
            seg_y1 = seg_y1,
            seg_x2 = seg_x2,
            seg_y2 = seg_y2,
            line_a = line_a_unnormalized / line_norm,
            line_b = line_b_unnormalized / line_norm,
            line_c = line_c_unnormalized / line_norm,
            diff_x = (seg_x2 - seg_x1)[:,np.newaxis],
            diff_y = (seg_y2 - seg_y1)[:,np.newaxis],
        )


    @classmethod
    def union(cls: Type[LineSegmentsTypevar],
              l1: LineSegmentsTypevar, l2: LineSegmentsTypevar,
              ) -> LineSegmentsTypevar:
        return cls(
            seg_x1=np.concatenate([l1.seg_x1, l2.seg_x1]),
            seg_y1=np.concatenate([l1.seg_y1, l2.seg_y1]),
            seg_x2=np.concatenate([l1.seg_x2, l2.seg_x2]),
            seg_y2=np.concatenate([l1.seg_y2, l2.seg_y2]),
            line_a=np.concatenate([l1.line_a, l2.line_a]),
            line_b=np.concatenate([l1.line_b, l2.line_b]),
            line_c=np.concatenate([l1.line_c, l2.line_c]),
            diff_x=np.concatenate([l1.diff_x, l2.diff_x], axis=0),
            diff_y=np.concatenate([l1.diff_y, l2.diff_y], axis=0),
        )

    @classmethod
    def empty(cls: Type[LineSegmentsTypevar]) -> LineSegmentsTypevar:
        return cls(
            seg_x1=np.zeros(shape=(0,)),
            seg_y1=np.zeros(shape=(0,)),
            seg_x2=np.zeros(shape=(0,)),
            seg_y2=np.zeros(shape=(0,)),
            line_a=np.zeros(shape=(0,)),
            line_b=np.zeros(shape=(0,)),
            line_c=np.zeros(shape=(0,)),
            diff_x=np.zeros(shape=(0,1)),
            diff_y=np.zeros(shape=(0,1)),
        )


    def distance(self,
                 px_flat: np.ndarray, py_flat: np.ndarray,
                 ) -> np.ndarray:
        # taken from https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        #
        # each segment is defined by the line
        # from (x1[i], y1[i]) to (x2[i], y2[i])
        #
        # each line segment is contained in a line of the form
        # a*x + b*y + c = 0 where a^2 + b^2 = 1
        #
        # result[j] is the distance of (px[j], py[j]) to the nearest segment
        px = px_flat[:,np.newaxis]
        py = py_flat[:,np.newaxis]
        dist_to_p1 = np.sqrt(np.square(self.seg_x1 - px) + np.square(self.seg_y1 - py))
        dist_to_p2 = np.sqrt(np.square(self.seg_x2 - px) + np.square(self.seg_y2 - py))
        dist_to_line = np.abs(self.line_a * px + self.line_b * py + self.line_c)
        clo_x = self.line_b * (self.line_b * px - self.line_a * py) - self.line_a * self.line_c
        clo_y = self.line_a * (self.line_a * py - self.line_b * px) - self.line_b * self.line_c
        is_in_seg_x = (self.seg_x1 - EPSILON <= clo_x) & (clo_x <= self.seg_x2 + EPSILON) \
            | (self.seg_x2 - EPSILON <= clo_x) & (clo_x <= self.seg_x1 + EPSILON)
        is_in_seg_y = (self.seg_y1 - EPSILON <= clo_y) & (clo_y <= self.seg_y2 + EPSILON) \
            | (self.seg_y2 - EPSILON <= clo_y) & (clo_y <= self.seg_y1 + EPSILON)
        is_in_segment = is_in_seg_x & is_in_seg_y
        dist_to_segment = np.where(
            is_in_segment,
            dist_to_line,
            np.minimum(dist_to_p1, dist_to_p2),
        )
        return np.amin(dist_to_segment, axis=1)

class State(NamedTuple):
    x: float
    y: float
    V: float
    theta: float
    obstacle_x: float
    obstacle_y: float
    lines: LineSegments

class Polyhedron:
    A: np.ndarray
    b: np.ndarray

    def __init__(self, A: np.ndarray, b: np.ndarray) -> None:
        (m1, d) = A.shape
        (m2,) = b.shape
        assert m1 == m2
        assert d == 4
        self.A = A
        self.b = b

    def contains(self, st: State) -> bool:
        vec = np.array([st.x, st.y, st.V, st.theta])
        return np.all(self.A @ vec <= self.b)

class F110ObstacleMode(Mode[State]):
    num_lidar_rays: int
    use_beta: bool
    use_throttle: bool
    rng: np.random.Generator

    obstacle_polygons: List[List[FloatPair]]
    obstacle_paths: List[List[FloatPair]]
    obstacle_x_low: float
    obstacle_y_low: float
    obstacle_x_high: float
    obstacle_y_high: float

    start_x: float
    start_y: float
    start_V: float
    start_theta: float
    start_x_noise: float
    start_y_noise: float
    start_V_noise: float
    start_theta_noise: float

    goal_x: float
    goal_y: float
    goal_theta: float
    goal_region: Polyhedron

    static_lines: LineSegments

    center_reward_region: Polyhedron
    center_reward_lines: LineSegments

    def __init__(self,
                 name: str,
                 static_polygons: Iterable[Iterable[FloatPair]],
                 static_paths: Iterable[Iterable[FloatPair]],
                 obstacle_polygons: Iterable[Iterable[FloatPair]],
                 obstacle_paths: Iterable[Iterable[FloatPair]],
                 obstacle_x_low: float,
                 obstacle_y_low: float,
                 obstacle_x_high: float,
                 obstacle_y_high: float,
                 start_x: float,
                 start_y: float,
                 start_V: float,
                 start_theta: float,
                 start_x_noise: float,
                 start_y_noise: float,
                 start_V_noise: float,
                 start_theta_noise: float,
                 goal_x: float,
                 goal_y: float,
                 goal_theta: float,
                 goal_region: Polyhedron,
                 center_reward_region: Polyhedron,
                 center_reward_path: Tuple[FloatPair, FloatPair],
                 num_lidar_rays: int = 1081,
                 use_beta: bool = False,
                 use_throttle: bool = True,
                 rng: np.random.Generator = np.random.default_rng(),
                 ) -> None:
        self.num_lidar_rays = num_lidar_rays
        self.use_beta = use_beta
        self.use_throttle = use_throttle
        self.rng = rng

        self.obstacle_polygons = [list(p) for p in obstacle_polygons]
        self.obstacle_paths = [list(p) for p in obstacle_paths]
        self.obstacle_x_low = obstacle_x_low
        self.obstacle_y_low = obstacle_y_low
        self.obstacle_x_high = obstacle_x_high
        self.obstacle_y_high = obstacle_y_high

        self.start_x = start_x
        self.start_y = start_y
        self.start_V = start_V
        self.start_theta = start_theta
        self.start_x_noise = start_x_noise
        self.start_y_noise = start_y_noise
        self.start_V_noise = start_V_noise
        self.start_theta_noise = start_theta_noise

        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_theta = goal_theta
        self.goal_region = goal_region

        self.center_reward_region = center_reward_region
        self.center_reward_lines = LineSegments.from_polygons(
            polygons=[], paths=[center_reward_path],
        )

        self.static_lines = LineSegments.from_polygons(
            polygons=static_polygons, paths=static_paths,
        )

        super().__init__(
            name=name,
            action_space=spaces.Box(low=-1.0, high=1.0,
                                    shape=(2 if use_throttle else 1,), dtype=np.float32),
            observation_space=spaces.Box(low=0, high=LIDAR_RANGE,
                                         shape=(num_lidar_rays,), dtype=np.float32)
        )

    def compute_obstacle_lines(self,
                               obstacle_x: float,
                               obstacle_y: float,
                               ) -> LineSegments:
        return LineSegments.union(
            self.static_lines,
            LineSegments.from_polygons(
                polygons=add_to_tuples(self.obstacle_polygons, obstacle_x, obstacle_y),
                paths=add_to_tuples(self.obstacle_paths, obstacle_x, obstacle_y),
            )
        )

    def random_obstacle_pos(self) -> FloatPair:
        return self.rng.uniform(self.obstacle_x_low, self.obstacle_x_high), \
            self.rng.uniform(self.obstacle_y_low, self.obstacle_y_high)

    def reset(self) -> State:
        obstacle_x, obstacle_y = self.random_obstacle_pos()
        return State(
            x = self.rng.uniform(self.start_x - self.start_x_noise, self.start_x + self.start_x_noise),
            y = self.rng.uniform(self.start_y - self.start_y_noise, self.start_y + self.start_y_noise),
            V = self.rng.uniform(self.start_V - self.start_V_noise, self.start_V + self.start_V_noise),
            theta = self.rng.uniform(self.start_theta - self.start_theta_noise, self.start_theta + self.start_theta_noise),
            obstacle_x = obstacle_x, obstacle_y = obstacle_y,
            lines = self.compute_obstacle_lines(obstacle_x, obstacle_y),
        )

    def is_safe(self, st: State) -> bool:
        min_dist = st.lines.distance(np.array([st.x]), np.array([st.y]))
        assert min_dist.shape == (1,), f'incorrect shape {min_dist.shape}'
        return min_dist[0] > SAFE_DISTANCE

    def plot_halls(self, ax: mpl.axes.Axes, st: State) -> None:
        for i in range(st.lines.seg_x1.shape[0]):
            ax.add_line(mpl.lines.Line2D(
                [st.lines.seg_x1[i], st.lines.seg_x2[i]],
                [st.lines.seg_y1[i], st.lines.seg_y2[i]],
            ))

    def plot_reward_line(self, ax: mpl.axes.Axes) -> None:
        lines = self.center_reward_lines
        for i in range(lines.seg_x1.shape[0]):
            ax.add_line(mpl.lines.Line2D(
                [lines.seg_x1[i], lines.seg_x2[i]],
                [lines.seg_y1[i], lines.seg_y2[i]],
                color='r',
            ))

    def plot_lidar(self, ax: mpl.axes.Axes, st: State) -> None:
        scan = self._observation_fn(st)
        angle = np.linspace(
            st.theta - LIDAR_FIELD_OF_VIEW,
            st.theta + LIDAR_FIELD_OF_VIEW,
            self.num_lidar_rays,
        )
        ax.plot([st.x], [st.y], color='r', marker='.', linestyle='None')
        end_x = []
        end_y = []
        for i in range(self.num_lidar_rays):
            end_x.append(st.x + scan[i] * np.cos(angle[i]))
            end_y.append(st.y + scan[i] * np.sin(angle[i]))
        ax.plot(end_x, end_y, color='g', marker='.', linestyle='None')
            
    def plot_trajectory(self,
                        ax: mpl.axes.Axes,
                        trajectory: Iterable[State],
                        ) -> None:
        ax.plot(
            [st.x for st in trajectory],
            [st.y for st in trajectory],
            'r--',
        )

    def render(self, st: State) -> None:
        fig, ax = plt.subplots()
        self.plot_halls(ax, st)
        self.plot_lidar(ax, st)
        plt.show()

    def _step_fn(self, st: State, action: np.ndarray) -> State:
        delta = action[0] * MAX_TURNING_INPUT
        u = 0.5 * (action[1] + 1.0) * MAX_THROTTLE \
            if self.use_throttle else CONST_THROTTLE
        beta = np.arctan(CAR_CENTER_OF_MASS * np.tan(delta) / CAR_LENGTH) if self.use_beta else 0
        equilibrium_speed = CAR_MOTOR_CONST * np.maximum(u - HYSTERESIS_CONSTANT, 0)
        deviation_div_accel = (equilibrium_speed - st.V) / CAR_ACCEL_CONST
        V = (st.V - equilibrium_speed) * EKAT + equilibrium_speed
        d = deviation_div_accel * EKAT + equilibrium_speed * TIME_CHECK - deviation_div_accel
        if np.abs(delta) > 1e-5:
            dth_dd = np.cos(beta) * np.tan(delta) / CAR_LENGTH
            theta = dth_dd * d + st.theta
            x = (np.sin(theta + beta) - np.sin(st.theta + beta)) / dth_dd + st.x
            y = (np.cos(st.theta + beta) - np.cos(theta + beta)) / dth_dd + st.y
        else:
            theta = np.full_like(d, st.theta)
            x = np.cos(theta + beta) * d + st.x
            y = np.sin(theta + beta) * d + st.y
        seg_dists = st.lines.distance(x, y)
        i_eff = first_true(seg_dists <= SAFE_DISTANCE)
        # i_eff is -1 (the last element) if there are no unsafe points in the trajectory
        return State(
            x=x[i_eff], y=y[i_eff], V=V[i_eff], theta=theta[i_eff],
            obstacle_x=st.obstacle_x, obstacle_y=st.obstacle_y,
            lines=st.lines,
        )

    def _observation_fn(self, st: State) -> np.ndarray:
        theta = np.linspace(
            st.theta - LIDAR_FIELD_OF_VIEW,
            st.theta + LIDAR_FIELD_OF_VIEW,
            self.num_lidar_rays,
        )
        x_pt = st.lines.seg_x2[:,np.newaxis] - st.x
        y_pt = st.lines.seg_y2[:,np.newaxis] - st.y
        determinant = st.lines.diff_y * np.cos(theta) - st.lines.diff_x * np.sin(theta)
        c = (st.lines.diff_y * x_pt - st.lines.diff_x * y_pt) / determinant
        alpha = (y_pt * np.cos(theta) - x_pt * np.sin(theta)) / determinant
        segment_distances = np.where(
            (np.abs(determinant) > EPSILON) & (0 <= alpha) & (alpha <= 1) & (c > EPSILON),
            c, np.inf,
        )
        lidar_distances = np.amin(segment_distances, axis=0)
        return np.array(np.minimum(lidar_distances, LIDAR_RANGE))

    def _reward_fn(self, st0: State, action: np.ndarray, st1: State) -> float:
        if not self.is_safe(st1):
            return CRASH_REWARD

        reward = STEP_REWARD_GAIN

        reward += SPEED_GAIN * st1.V
        if st1.V < MINIMUM_ACCEPTABLE_SPEED:
            reward += LOW_SPEED_REWARD

        old_dist_to_goal = np.abs(self.goal_x - st0.x) + np.abs(self.goal_y - st0.y)
        new_dist_to_goal = np.abs(self.goal_x - st1.x) + np.abs(self.goal_y - st1.y)
        reward += PROGRESS_GAIN * (old_dist_to_goal - new_dist_to_goal)

        if self.center_reward_region.contains(st1):
            reward += MIDDLE_GAIN * self.center_reward_lines.distance(
                np.array([st1.x]), np.array([st1.y]),
            )[0]

        return reward

    def vectorize_state(self, st: State) -> np.ndarray:
        return np.array([st.x, st.y, st.V, st.theta, st.obstacle_x, st.obstacle_y])

    def state_from_vector(self, vec: np.ndarray) -> State:
        obstacle_x = vec[4]
        obstacle_y = vec[5]
        return State(
            x=vec[0], y=vec[1], V=vec[2], theta=vec[3],
            obstacle_x=obstacle_x, obstacle_y=obstacle_y,
            lines=self.compute_obstacle_lines(obstacle_x, obstacle_y),
        )

def make_obstacle(use_throttle: bool = True,
                  lidar_num_rays: int = 1081,
                  width: float = DEFAULT_HALL_WIDTH,
                  ) -> F110ObstacleMode:
    hhw = width / 2
    return F110ObstacleMode(
        name='f110_obstacle',
        static_polygons=[],
        static_paths=[
            [(-hhw,-20), (-hhw,-5), (-width, -3), (-width, 3), (-hhw, 5), (-hhw, 20)],
            [(hhw,-20), (hhw,-5), (width, -3), (width, 3), (hhw, 5), (hhw, 20)],
        ],
        obstacle_polygons=[[(0.5, 0.5), (0.5, -0.5), (-0.5, -0.5), (-0.5, 0.5)]],
        obstacle_paths=[],
        obstacle_x_low=-hhw,
        obstacle_x_high=hhw,
        obstacle_y_low=-1.0,
        obstacle_y_high=1.0,
        start_x=0,
        start_y=-11,
        start_V=2.5,
        start_theta=float(np.radians(90)),
        start_x_noise=0.2,
        start_y_noise=0.2,
        start_V_noise=2.5,
        start_theta_noise = float(np.radians(30)),
        goal_x=0,
        goal_y=15,
        goal_theta=float(np.radians(90)),
        goal_region=Polyhedron(
            # y >= 14.5
            # x >= -hhw
            # x <= hhw
            A=np.array([
                [0, -1, 0, 0],
                [-1, 0, 0, 0],
                [1, 0, 0, 0],
            ]),
            b=np.array([-14.5, hhw, hhw]),
        ),
        center_reward_region=Polyhedron(
            # y >= 5
            # x >= -hhw
            # x <= hhw
            A=np.array([
                [0, -1, 0, 0],
                [-1, 0, 0, 0],
                [1, 0, 0, 0],
            ]),
            b=np.array([-5, hhw, hhw]),
        ),
        center_reward_path=((0, 0), (0, 20)),
    )

def make_obstacle_right(use_throttle: bool = True,
                        lidar_num_rays: int = 1081,
                        width: float = DEFAULT_HALL_WIDTH,
                        ) -> F110ObstacleMode:
    hhw = width / 2
    return F110ObstacleMode(
        name='f110_obstacle_right',
        static_polygons=[],
        static_paths=[
            [(-hhw,-20), (-hhw,-5), (-width, -3), (-width, 3), (-hhw, 5), (-hhw, 20)],
            [(hhw,-20), (hhw,-5), (width, -3), (width, 3), (hhw, 5), (hhw, 20)],
        ],
        obstacle_polygons=[[(0.5, 0.5), (0.5, -0.5), (-0.5, -0.5), (-0.5, 0.5)]],
        obstacle_paths=[],
        obstacle_x_low=0,
        obstacle_x_high=hhw,
        obstacle_y_low=-1.0,
        obstacle_y_high=1.0,
        start_x=0,
        start_y=-11,
        start_V=2.5,
        start_theta=float(np.radians(90)),
        start_x_noise=0.2,
        start_y_noise=0.2,
        start_V_noise=2.5,
        start_theta_noise = float(np.radians(30)),
        goal_x=0,
        goal_y=15,
        goal_theta=float(np.radians(90)),
        goal_region=Polyhedron(
            # y >= 14.5
            # x >= -hhw
            # x <= hhw
            A=np.array([
                [0, -1, 0, 0],
                [-1, 0, 0, 0],
                [1, 0, 0, 0],
            ]),
            b=np.array([-14.5, hhw, hhw]),
        ),
        center_reward_region=Polyhedron(
            # y >= 5
            # x >= -hhw
            # x <= hhw
            A=np.array([
                [0, -1, 0, 0],
                [-1, 0, 0, 0],
                [1, 0, 0, 0],
            ]),
            b=np.array([-5, hhw, hhw]),
        ),
        center_reward_path=((0, 0), (0, 20)),
    )

def make_obstacle_left(use_throttle: bool = True,
                       lidar_num_rays: int = 1081,
                       width: float = DEFAULT_HALL_WIDTH,
                       ) -> F110ObstacleMode:
    hhw = width / 2
    return F110ObstacleMode(
        name='f110_obstacle_left',
        static_polygons=[],
        static_paths=[
            [(-hhw,-20), (-hhw,-5), (-width, -3), (-width, 3), (-hhw, 5), (-hhw, 20)],
            [(hhw,-20), (hhw,-5), (width, -3), (width, 3), (hhw, 5), (hhw, 20)],
        ],
        obstacle_polygons=[[(0.5, 0.5), (0.5, -0.5), (-0.5, -0.5), (-0.5, 0.5)]],
        obstacle_paths=[],
        obstacle_x_low=-hhw,
        obstacle_x_high=0,
        obstacle_y_low=-1.0,
        obstacle_y_high=1.0,
        start_x=0,
        start_y=-11,
        start_V=2.5,
        start_theta=float(np.radians(90)),
        start_x_noise=0.2,
        start_y_noise=0.2,
        start_V_noise=2.5,
        start_theta_noise = float(np.radians(30)),
        goal_x=0,
        goal_y=15,
        goal_theta=float(np.radians(90)),
        goal_region=Polyhedron(
            # y >= 14.5
            # x >= -hhw
            # x <= hhw
            A=np.array([
                [0, -1, 0, 0],
                [-1, 0, 0, 0],
                [1, 0, 0, 0],
            ]),
            b=np.array([-14.5, hhw, hhw]),
        ),
        center_reward_region=Polyhedron(
            # y >= 5
            # x >= -hhw
            # x <= hhw
            A=np.array([
                [0, -1, 0, 0],
                [-1, 0, 0, 0],
                [1, 0, 0, 0],
            ]),
            b=np.array([-5, hhw, hhw]),
        ),
        center_reward_path=((0, 0), (0, 20)),
    )
