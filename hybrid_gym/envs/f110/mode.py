import numpy as np
import matplotlib.pyplot as plt
import math
import enum

from gym import spaces
from hybrid_gym.model import Mode
from typing import List, Iterable, Tuple, Optional, NamedTuple

# hallway properties
DEFAULT_HALL_WIDTH: float = 1.5

# car parameters
CAR_LENGTH: float = .45  # in m
CAR_CENTER_OF_MASS: float = .225  # from rear of car (m)
CAR_DECEL_CONST: float = .4
CAR_ACCEL_CONST: float = 1.633  # estimated from data
CAR_MOTOR_CONST: float = 0.2  # estimated from data
HYSTERESIS_CONSTANT: float = 4
MAX_TURNING_INPUT: float = 20  # in degrees

# lidar parameter
LIDAR_RANGE: float = 5  # in m

# safety parameter
SAFE_DISTANCE: float = 0.1  # in m

# default throttle if left unspecified
CONST_THROTTLE: float = 16
MAX_THROTTLE: float = 30  # just used to compute maximum possible velocity

# training parameters
MINIMUM_ACCEPTABLE_SPEED: float = 1.5
LOW_SPEED_REWARD: float = -20
SPEED_GAIN: float = 0
STEP_REWARD_GAIN: float = 5
INPUT_REWARD_GAIN: float = 0
CRASH_REWARD: float = -100
MIDDLE_REWARD_GAIN: float = -3
HEADING_GAIN: float = -3
MOVE_FORWARD_GAIN: float = 10
REGION3_ENTER_GAIN: float = 0  # 100

# direction parameters


class Direction(enum.Enum):
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3


class State(NamedTuple):
    car_dist_s: float
    car_dist_f: float
    car_V: float
    car_heading: float
    car_global_x: float
    car_global_y: float
    car_global_heading: float
    direction: Direction
    curHall: int
    cur_hall_heading: float
    outer_x: float
    outer_y: float
    inner_x: float
    inner_y: float
    missing_indices: List[int]


class F110Mode(Mode[State]):
    hallWidths: List[float]
    hallLengths: List[float]
    numHalls: int
    turns: List[float]
    state_feedback: bool
    init_car_dist_s: float
    init_car_dist_f: float
    init_car_heading: float
    init_car_V: float
    time_step: float
    lidar_field_of_view: float
    use_throttle: bool
    lidar_num_rays: int
    lidar_noise: float
    lidar_missing_in_turn_only: bool
    cur_lidar_missing_rays: int
    init_outer_x: float
    init_outer_y: float
    init_inner_x: float
    init_inner_y: float

    def __init__(self,
                 name: str,
                 hallWidths: Iterable[float],
                 hallLengths: Iterable[float],
                 turns: Iterable[float],
                 init_car_dist_s: float,
                 init_car_dist_f: float,
                 init_car_heading: float,
                 init_car_V: float,
                 time_step: float,
                 lidar_field_of_view: float,
                 use_throttle: bool,
                 lidar_num_rays: int,
                 lidar_noise: float = 0,
                 lidar_missing_rays: int = 0,
                 lidar_missing_in_turn_only: bool = False,
                 state_feedback: bool = False
                 ) -> None:

        # hallway parameters
        self.hallWidths = list(hallWidths)
        self.hallLengths = list(hallLengths)
        self.numHalls = len(self.hallWidths)
        self.turns = list(turns)
        # self.curHall = 0
        # self.in_region3 = False
        # self.in_region3_1m = False
        # self.in_region3_2m = False
        # self.in_region3_3m = False

        # observation parameter
        self.state_feedback = state_feedback

        # car relative states
        # self.car_dist_s = car_dist_s
        # self.car_dist_f = car_dist_f
        # self.car_V = car_V
        # self.car_heading = car_heading

        # car global states
        # self.car_global_x = -self.hallWidths[0] / 2.0 + self.car_dist_s
        # if self.turns[0] > 0:
        #    self.car_global_x = -self.car_global_x

        # self.car_global_y = self.hallLengths[0] / 2.0 - car_dist_f
        # self.car_global_heading = self.car_heading + np.pi / 2 #first hall goes "up" by default
        # self.direction = UP

        # car initial conditions (used in reset)
        self.init_car_dist_s = init_car_dist_s
        self.init_car_dist_f = init_car_dist_f
        self.init_car_heading = init_car_heading
        self.init_car_V = init_car_V

        # step parameters
        self.time_step = time_step

        # storage
        # self.allX = []
        # self.allY = []
        # self.allX.append(self.car_global_x)
        # self.allY.append(self.car_global_y)

        self.use_throttle = use_throttle

        # lidar setup
        self.lidar_field_of_view = lidar_field_of_view
        self.lidar_num_rays = lidar_num_rays

        self.lidar_noise = lidar_noise
        # self.total_lidar_missing_rays = lidar_missing_rays

        self.lidar_missing_in_turn_only = lidar_missing_in_turn_only

        self.cur_num_missing_rays = lidar_missing_rays
        # self.missing_indices = np.random.choice(self.lidar_num_rays, self.cur_num_missing_rays)

        # coordinates of two corners in turn
        cur_hall_heading = np.pi/2
        next_heading = cur_hall_heading + self.turns[0]
        if next_heading > np.pi:
            next_heading -= 2 * np.pi
        elif next_heading < -np.pi:
            next_heading += 2 * np.pi

        reverse_cur_heading = cur_hall_heading - np.pi

        if self.turns[0] < 0:
            self.init_outer_x = -self.hallWidths[0]/2.0
            self.init_outer_y = self.hallLengths[0]/2.0

        else:
            self.init_outer_x = self.hallWidths[0]/2.0
            self.init_outer_y = self.hallLengths[0]/2.0

        out_wall_proj_length = np.abs(self.hallWidths[0] / np.sin(self.turns[0]))
        proj_point_x = self.init_outer_x + np.cos(next_heading) * out_wall_proj_length
        proj_point_y = self.init_outer_y + np.sin(next_heading) * out_wall_proj_length

        in_wall_proj_length = np.abs(self.hallWidths[1] / np.sin(self.turns[0]))
        self.init_inner_x = proj_point_x + np.cos(reverse_cur_heading) * in_wall_proj_length
        self.init_inner_y = proj_point_y + np.sin(reverse_cur_heading) * in_wall_proj_length

        # self.init_outer_x = self.outer_x
        # self.init_outer_y = self.outer_y
        # self.init_inner_x = self.inner_x
        # self.init_inner_y = self.inner_y

        # parameters needed for consistency with gym environments
        if state_feedback:
            # self.obs_low = np.array([0, 0, 0, -np.pi])
            # self.obs_high = np.array([max(hallLengths), max(hallLengths), CAR_MOTOR_CONST *
            #   (MAX_THROTTLE - HYSTERESIS_CONSTANT), np.pi])

            # self.obs_low = np.array([0, 0, -np.pi])
            # self.obs_high = np.array([max(hallLengths), max(hallLengths), np.pi])

            obs_low = np.array([0, 0, -2*max(hallWidths), -2*max(hallWidths), -np.pi])
            obs_high = np.array([LIDAR_RANGE, LIDAR_RANGE, LIDAR_RANGE, LIDAR_RANGE, np.pi])

        else:
            obs_low = np.zeros(self.lidar_num_rays, )
            obs_high = LIDAR_RANGE * np.ones(self.lidar_num_rays, )

        num_actions = 2 if self.use_throttle else 1

        super().__init__(
            name=name,
            action_space=spaces.Box(low=-1.0, high=1.0,
                                    shape=(num_actions,), dtype=np.float32),
            observation_space=spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        )

        # self._max_episode_steps = episode_length

    # this is a limited-support function for setting the car state in the first hallway
    def set_state_local(self, x: float, y: float, theta: float, old_st: State,
                        v: Optional[float] = None) -> State:

        car_dist_s = x
        car_dist_f = y
        car_heading = theta
        if v is not None:
            car_V = v
        else:
            car_V = old_st.car_V

        car_global_x = x - self.hallWidths[0]/2
        if self.turns[0] > 0:
            car_global_x = -car_global_x
        car_global_y = -y + self.hallLengths[0]/2
        car_global_heading = theta + np.pi / 2
        direction = Direction.UP

        # test if in Region 3
        if y > self.hallLengths[0] - LIDAR_RANGE:

            direction = Direction.RIGHT

            temp = x
            car_dist_s = self.hallLengths[0] - y
            car_dist_f = temp
            car_heading = theta - np.pi / 2

            if car_heading < - np.pi:
                car_heading = car_heading + 2 * np.pi

        if car_global_heading > np.pi:
            car_global_heading = car_global_heading - 2 * np.pi

        return State(
            car_dist_s=car_dist_s,
            car_dist_f=car_dist_f,
            car_V=car_V,
            car_heading=car_heading,
            car_global_x=car_global_x,
            car_global_y=car_global_y,
            car_global_heading=car_global_heading,
            direction=direction,
            curHall=0,
            cur_hall_heading=0.5*np.pi,
            outer_x=self.init_outer_x,
            outer_y=self.init_outer_y,
            inner_x=self.init_inner_x,
            inner_y=self.init_inner_y,
            missing_indices=old_st.missing_indices,
        )

    # this is a limited-support function for setting the car state in the first hallway
    def set_state_global(self, x: float, y: float, theta: float, old_st: State) -> State:

        car_dist_s = x + self.hallWidths[0]/2
        car_dist_f = -y + self.hallLengths[0]/2
        car_heading = theta - np.pi / 2

        car_global_x = x
        car_global_y = y
        car_global_heading = theta

        direction = Direction.UP

        # test if in Region 3
        if y > self.hallLengths[0] - LIDAR_RANGE:

            direction = Direction.RIGHT

            temp = x
            car_dist_s = self.hallLengths[0] - y
            car_dist_f = temp
            car_heading = theta - np.pi / 2

            if car_heading < - np.pi:
                car_heading = old_st.car_heading + 2 * np.pi

        if car_global_heading > np.pi:
            car_global_heading = old_st.car_global_heading - 2 * np.pi

        return State(
            car_dist_s=car_dist_s,
            car_dist_f=car_dist_f,
            car_V=old_st.car_V,
            car_heading=car_heading,
            car_global_x=car_global_x,
            car_global_y=car_global_y,
            car_global_heading=car_global_heading,
            direction=direction,
            curHall=0,
            cur_hall_heading=0.5*np.pi,
            outer_x=self.init_outer_x,
            outer_y=self.init_outer_y,
            inner_x=self.init_inner_x,
            inner_y=self.init_inner_y,
            missing_indices=old_st.missing_indices,
        )

    def reset(self,
              side_pos: Optional[float] = None,
              pos_noise: float = 0.2,
              heading_noise: float = 0.1,
              front_pos_noise: float = 0.2
              ) -> State:
        # self.curHall = 0

        car_dist_s = self.init_car_dist_s + np.random.uniform(-pos_noise, pos_noise)

        if side_pos is not None:
            car_dist_s = side_pos

        car_dist_f = self.init_car_dist_f + np.random.uniform(-front_pos_noise, front_pos_noise)
        car_V = self.init_car_V
        car_heading = self.init_car_heading + np.random.uniform(-heading_noise, heading_noise)

        car_global_x = -self.hallWidths[0] / 2.0 + car_dist_s
        if self.turns[0] > 0:
            car_global_x = -car_global_x

        car_global_y = self.hallLengths[0] / 2.0 - car_dist_f

        car_global_heading = car_heading + np.pi / 2  # first hall goes "up" by default
        direction = Direction.UP

        # missing_indices = np.random.choice(self.lidar_num_rays, self.cur_num_missing_rays)

        # self.cur_step = 0

        # outer_x = self.init_outer_x
        # outer_y = self.init_outer_y
        # inner_x = self.init_inner_x
        # inner_y = self.init_inner_y

        # self.in_region3 = False
        # self.in_region3_1m = False
        # self.in_region3_2m = False
        # self.in_region3_3m = False

        # self.allX = []
        # self.allY = []
        # self.allX.append(self.car_global_x)
        # self.allY.append(self.car_global_y)

        # if self.state_feedback:
        #    #return np.array([self.car_dist_s, self.car_dist_f, self.car_V, self.car_heading])

        #    corner_dist = np.sqrt((self.outer_x - self.inner_x) ** 2 +
        #               (self.outer_y - self.inner_y) ** 2)
        #    wall_dist = np.sqrt(corner_dist ** 2 -
        #               self.hallWidths[(self.curHall+1)%self.numHalls] ** 2)

        #    dist_f = self.car_dist_f
        #    dist_f_inner = self.car_dist_f - wall_dist

        #    if dist_f > LIDAR_RANGE:
        #        dist_f = LIDAR_RANGE
        #    if dist_f_inner > LIDAR_RANGE:
        #        dist_f_inner = LIDAR_RANGE

        #    if self.turns[self.curHall] <= 0:
        #        return np.array([self.car_dist_s, self.hallWidths[self.curHall] - self.car_dist_s,\
        #                         dist_f, dist_f_inner, self.car_heading])
        #    else:
        #        return np.array([self.hallWidths[self.curHall] - self.car_dist_s, self.car_dist_s,\
        #                         dist_f_inner, dist_f, self.car_heading])
        # else:
        #    return self.scan_lidar()

        self.init_state = State(
            car_dist_s=car_dist_s,
            car_dist_f=car_dist_f,
            car_V=car_V,
            car_heading=car_heading,
            car_global_x=car_global_x,
            car_global_y=car_global_y,
            car_global_heading=car_global_heading,
            direction=direction,
            curHall=0,
            cur_hall_heading=0.5*np.pi,
            outer_x=self.init_outer_x,
            outer_y=self.init_outer_y,
            inner_x=self.init_inner_x,
            inner_y=self.init_inner_y,
            missing_indices=np.random.choice(self.lidar_num_rays, self.cur_num_missing_rays),
        )
        return self.init_state

    # NB: Mode switches are handled in the step function
    # x := [s, f, V, theta_local, x, y, theta_global]
    def bicycle_dynamics_ode(self, x, t, u, delta, turn):

        if turn < 0:  # right turn
            # -V * sin(theta_local + beta)
            dsdt = -x[2] * np.sin(x[3] + np.arctan(CAR_CENTER_OF_MASS * np.tan(delta) / CAR_LENGTH))
        else:
            # V * sin(theta_local + beta)
            dsdt = x[2] * np.sin(x[3] + np.arctan(CAR_CENTER_OF_MASS * np.tan(delta) / CAR_LENGTH))

        # -V * cos(theta_local + beta)
        dfdt = -x[2] * np.cos(x[3] + np.arctan(CAR_CENTER_OF_MASS * np.tan(delta) / CAR_LENGTH))

        if u > HYSTERESIS_CONSTANT:
            # a * u - V
            dVdt = CAR_ACCEL_CONST * CAR_MOTOR_CONST * \
                (u - HYSTERESIS_CONSTANT) - CAR_ACCEL_CONST * x[2]
        else:
            dVdt = - CAR_ACCEL_CONST * x[2]

        dtheta_ldt = x[2] * np.cos(np.arctan(
            CAR_CENTER_OF_MASS * np.tan(delta) / CAR_LENGTH)) * np.tan(delta) / CAR_LENGTH

        # V * cos(theta_global + beta)
        dxdt = x[2] * np.cos(x[6] + np.arctan(CAR_CENTER_OF_MASS * np.tan(delta) / CAR_LENGTH))

        # V * sin(theta_global + beta)
        dydt = x[2] * np.sin(x[6] + np.arctan(CAR_CENTER_OF_MASS * np.tan(delta) / CAR_LENGTH))

        # V * cos(beta) * tan(delta) / l
        dtheta_gdt = x[2] * np.cos(np.arctan(
            CAR_CENTER_OF_MASS * np.tan(delta) / CAR_LENGTH)) * np.tan(delta) / CAR_LENGTH

        dXdt = [dsdt, dfdt, dVdt, dtheta_ldt, dxdt, dydt, dtheta_gdt]

        return dXdt

    # NB: Mode switches are handled in the step function
    # x := [s, f, V, theta_local, x, y, theta_global]
    def bicycle_dynamics_ode_no_beta(self, x, t, u, delta, turn):

        if turn < 0:  # right turn
            # -V * sin(theta_local)
            dsdt = -x[2] * np.sin(x[3])
        else:
            # V * sin(theta_local)
            dsdt = x[2] * np.sin(x[3])

        # -V * cos(theta_local)
        dfdt = -x[2] * np.cos(x[3])

        if u > HYSTERESIS_CONSTANT:
            # a * u - V
            dVdt = CAR_ACCEL_CONST * CAR_MOTOR_CONST * \
                (u - HYSTERESIS_CONSTANT) - CAR_ACCEL_CONST * x[2]
        else:
            dVdt = - CAR_ACCEL_CONST * x[2]

        # V * tan(delta) / l
        dtheta_ldt = x[2] * np.tan(delta) / CAR_LENGTH

        # V * cos(theta_global)
        dxdt = x[2] * np.cos(x[6])

        # V * sin(theta_global)
        dydt = x[2] * np.sin(x[6])

        # V * tan(delta) / l
        dtheta_gdt = x[2] * np.tan(delta) / CAR_LENGTH

        dXdt = [dsdt, dfdt, dVdt, dtheta_ldt, dxdt, dydt, dtheta_gdt]

        return dXdt

    def bicycle_dynamics_clf(self,
                             st: State,
                             delta: float,
                             throttle: float,
                             is_right_turn: bool,
                             use_beta: bool,
                             ) -> np.ndarray:
        ekat = np.exp(-CAR_ACCEL_CONST * self.time_step)
        beta = np.arctan(CAR_CENTER_OF_MASS * np.tan(delta) / CAR_LENGTH) if use_beta else 0.0
        equilibrium_speed = CAR_MOTOR_CONST * np.maximum(throttle - HYSTERESIS_CONSTANT, 0)
        deviation_div_accel = (equilibrium_speed - st.car_V) / CAR_ACCEL_CONST
        car_V = (st.car_V - equilibrium_speed) * ekat + equilibrium_speed
        d = deviation_div_accel * ekat + equilibrium_speed * self.time_step - deviation_div_accel
        if np.abs(delta) > 1e-5:
            dth_dd = np.cos(beta) * np.tan(delta) / CAR_LENGTH
            car_heading = dth_dd * d + st.car_heading
            change_s = (np.cos(st.car_heading + beta) - np.cos(car_heading + beta)) / dth_dd
            if is_right_turn:
                change_s *= -1
            car_dist_s = change_s + st.car_dist_s
            car_dist_f = (np.sin(st.car_heading + beta) -
                          np.sin(car_heading + beta)) / dth_dd + st.car_dist_f
            car_global_heading = dth_dd * d + st.car_global_heading
            car_global_x = (np.sin(car_global_heading + beta) -
                            np.sin(st.car_global_heading + beta)) / dth_dd + st.car_global_x
            car_global_y = (np.cos(st.car_global_heading + beta) -
                            np.cos(car_global_heading + beta)) / dth_dd + st.car_global_y
        else:
            car_heading = st.car_heading
            change_s = np.sin(car_heading) * d
            if is_right_turn:
                change_s *= -1
            car_dist_s = change_s + st.car_dist_s
            car_dist_f = -np.cos(car_heading) * d + st.car_dist_f
            car_global_heading = st.car_global_heading
            car_global_x = np.cos(car_global_heading) * d + st.car_global_x
            car_global_y = np.sin(car_global_heading) * d + st.car_global_y
        return np.array([car_dist_s, car_dist_f, car_V, car_heading,
                         car_global_x, car_global_y, car_global_heading])

    def _step_fn(self, st: State, action: np.ndarray) -> State:
        delta = MAX_TURNING_INPUT * action[0]
        throttle = 0.5 * (action[1] + 1.0) * MAX_THROTTLE \
            if self.use_throttle else CONST_THROTTLE
        return self.helper_step(st, delta, throttle)

    def helper_step(self,
                    st: State,
                    delta: float,
                    throttle: float,
                    x_noise: float = 0,
                    y_noise: float = 0,
                    v_noise: float = 0,
                    theta_noise: float = 0
                    ) -> State:
        # self.cur_step += 1

        # Constrain turning input
        if delta > MAX_TURNING_INPUT:
            delta = MAX_TURNING_INPUT

        if delta < -MAX_TURNING_INPUT:
            delta = -MAX_TURNING_INPUT

        # simulate dynamics
        x0 = [st.car_dist_s, st.car_dist_f, st.car_V, st.car_heading,
              st.car_global_x, st.car_global_y, st.car_global_heading]
        # t = [0, self.time_step]

        # new_x = odeint(self.bicycle_dynamics, x0, t, args=(throttle, delta * np.pi / 180,
        #       self.turns[st.curHall],))
        # new_x = odeint(self.bicycle_dynamics_no_beta, x0, t, args=(
        #    throttle, delta * np.pi / 180, self.turns[st.curHall],))

        # new_x = new_x[1]
        new_x = self.bicycle_dynamics_clf(
            st=st, delta=np.float(np.radians(delta)), throttle=throttle,
            is_right_turn=(self.turns[st.curHall] < 0), use_beta=False,
        )

        # add noise
        x_added_noise = x_noise * (2 * np.random.random() - 1)
        y_added_noise = y_noise * (2 * np.random.random() - 1)
        v_added_noise = v_noise * (2 * np.random.random() - 1)
        # theta_added_noise = theta_noise * (2 * np.random.random() - 1)
        theta_added_noise = theta_noise * (np.random.random())

        new_x[0] = new_x[0] + x_added_noise

        if st.direction == Direction.UP and self.turns[st.curHall] == -np.pi/2\
           or st.direction == Direction.DOWN and self.turns[st.curHall] == np.pi/2:
            new_x[4] = new_x[4] + x_added_noise

        elif st.direction == Direction.DOWN and self.turns[st.curHall] == -np.pi/2\
                or st.direction == Direction.UP and self.turns[st.curHall] == np.pi/2:
            new_x[4] = new_x[4] - x_added_noise

        elif st.direction == Direction.RIGHT and self.turns[st.curHall] == -np.pi/2\
                or st.direction == Direction.LEFT and self.turns[st.curHall] == np.pi/2:
            new_x[4] = new_x[4] - y_added_noise

        elif st.direction == Direction.LEFT and self.turns[st.curHall] == -np.pi/2\
                or st.direction == Direction.RIGHT and self.turns[st.curHall] == np.pi/2:
            new_x[4] = new_x[4] + y_added_noise

        new_x[1] = new_x[1] + y_added_noise

        if st.direction == Direction.UP and self.turns[st.curHall] == -np.pi/2\
           or st.direction == Direction.DOWN and self.turns[st.curHall] == np.pi/2:
            new_x[5] = new_x[5] - y_added_noise

        elif st.direction == Direction.DOWN and self.turns[st.curHall] == -np.pi/2\
                or st.direction == Direction.UP and self.turns[st.curHall] == np.pi/2:
            new_x[5] = new_x[5] + y_added_noise

        elif st.direction == Direction.RIGHT and self.turns[st.curHall] == -np.pi/2\
                or st.direction == Direction.LEFT and self.turns[st.curHall] == np.pi/2:
            new_x[5] = new_x[5] - x_added_noise

        elif st.direction == Direction.LEFT and self.turns[st.curHall] == -np.pi/2\
                or st.direction == Direction.RIGHT and self.turns[st.curHall] == np.pi/2:
            new_x[5] = new_x[5] + x_added_noise

        new_x[2] = new_x[2] + v_added_noise

        # new_x[3] = new_x[3] + theta_added_noise
        # new_x[6] = new_x[6] + theta_added_noise

        # NB: The heading noise only affects heading in the direction
        # of less change

        if new_x[3] < x0[3]:
            new_x[3] = new_x[3] + theta_added_noise
            new_x[6] = new_x[6] + theta_added_noise
        else:
            new_x[3] = new_x[3] - theta_added_noise
            new_x[6] = new_x[6] - theta_added_noise
        # end of adding noise

        # delta_s = new_x[0] - st.car_dist_s
        # delta_f = st.car_dist_f - new_x[1]

        # compute delta along the 2nd hallway
        # old_s = st.car_dist_s
        # old_f = st.car_dist_f

        (car_dist_s, car_dist_f, car_V, car_heading,
         car_global_x, car_global_y, car_global_heading) =\
            new_x[0], new_x[1], new_x[2], new_x[3], new_x[4], new_x[5], new_x[6]

        if car_heading > np.pi:
            car_heading -= 2*np.pi
        elif car_heading < -np.pi:
            car_heading += 2*np.pi

        # terminal = False

        # Compute reward (moved to separate function)

        # if self.cur_step == self.episode_length:
        #    terminal = True

        curHall = st.curHall
        cur_hall_heading = st.cur_hall_heading
        outer_x = st.outer_x
        outer_y = st.outer_y
        inner_x = st.inner_x
        inner_y = st.inner_y

        # Check if a mode switch in the world has changed
        if car_dist_s > LIDAR_RANGE:

            # update global hall heading
            cur_hall_heading = st.cur_hall_heading + self.turns[st.curHall]
            if cur_hall_heading > np.pi:
                cur_hall_heading -= 2 * np.pi
            elif cur_hall_heading < -np.pi:
                cur_hall_heading += 2 * np.pi

            flip_sides = False

            # rightish turn
            if self.turns[st.curHall] < 0:

                # update corner coordinates
                if self.turns[(st.curHall+1) % self.numHalls] < 0:
                    flip_sides = False

                else:
                    flip_sides = True

                if st.direction == Direction.UP:
                    direction = Direction.RIGHT
                elif st.direction == Direction.RIGHT:
                    direction = Direction.DOWN
                elif st.direction == Direction.DOWN:
                    direction = Direction.LEFT
                elif st.direction == Direction.LEFT:
                    direction = Direction.UP

            else:  # left turn
                # update corner coordinates
                if self.turns[(st.curHall+1) % self.numHalls] > 0:
                    flip_sides = False

                else:
                    flip_sides = True

                if st.direction == Direction.UP:
                    direction = Direction.LEFT
                elif st.direction == Direction.RIGHT:
                    direction = Direction.UP
                elif st.direction == Direction.DOWN:
                    direction = Direction.RIGHT
                elif st.direction == Direction.LEFT:
                    direction = Direction.DOWN

            # update local car states
            st1 = State(
                car_dist_s=car_dist_s,
                car_dist_f=car_dist_f,
                car_V=car_V,
                car_heading=car_heading,
                car_global_x=car_global_x,
                car_global_y=car_global_y,
                car_global_heading=car_global_heading,
                direction=direction,
                curHall=curHall,
                cur_hall_heading=cur_hall_heading,
                outer_x=outer_x,
                outer_y=outer_y,
                inner_x=inner_x,
                inner_y=inner_y,
                missing_indices=st.missing_indices,
            )
            (car_dist_s, car_dist_f, car_heading) = self.next_car_states(st1, flip_sides)

            # update corner coordinates
            st2 = State(
                car_dist_s=car_dist_s,
                car_dist_f=car_dist_f,
                car_V=car_V,
                car_heading=car_heading,
                car_global_x=car_global_x,
                car_global_y=car_global_y,
                car_global_heading=car_global_heading,
                direction=direction,
                curHall=curHall,
                cur_hall_heading=cur_hall_heading,
                outer_x=outer_x,
                outer_y=outer_y,
                inner_x=inner_x,
                inner_y=inner_y,
                missing_indices=st.missing_indices,
            )
            (outer_x, outer_y, inner_x, inner_y) = self.next_corner_coordinates(st2, flip_sides)

            # update hall index
            curHall = st.curHall + 1  # next hallway
            # NB: this case deals with loops in the environment
            if curHall >= self.numHalls:
                curHall = 0
        else:
            direction = st.direction

        # self.allX.append(self.car_global_x)
        # self.allY.append(self.car_global_y)

        return State(
            car_dist_s=car_dist_s,
            car_dist_f=car_dist_f,
            car_V=car_V,
            car_heading=car_heading,
            car_global_x=car_global_x,
            car_global_y=car_global_y,
            car_global_heading=car_global_heading,
            direction=direction,
            curHall=curHall,
            cur_hall_heading=cur_hall_heading,
            outer_x=outer_x,
            outer_y=outer_y,
            inner_x=inner_x,
            inner_y=inner_y,
            missing_indices=st.missing_indices,
        )

    def _observation_fn(self, st: State) -> np.ndarray:
        if self.state_feedback:
            # return np.array([self.car_dist_s, self.car_dist_f, self.car_V, self.car_heading]),
            #           reward, terminal, -1

            if st.car_dist_s <= self.hallWidths[st.curHall]:

                corner_dist = np.sqrt((st.outer_x - st.inner_x) ** 2 +
                                      (st.outer_y - st.inner_y) ** 2)
                wall_dist = np.sqrt(corner_dist ** 2 -
                                    self.hallWidths[(st.curHall+1) % self.numHalls] ** 2)

                dist_s = st.car_dist_s
                dist_s2 = self.hallWidths[st.curHall] - dist_s
                dist_f = st.car_dist_f
                dist_f2 = dist_f - wall_dist
                car_heading = st.car_heading
            else:

                flip_sides = False
                (next_outer_x, next_outer_y, next_inner_x,
                 next_inner_y) = self.next_corner_coordinates(st, flip_sides)

                corner_dist = np.sqrt((next_outer_x - next_inner_x) ** 2 +
                                      (next_outer_y - next_inner_y) ** 2)
                wall_dist = np.sqrt(corner_dist ** 2 -
                                    self.hallWidths[(st.curHall+2) % self.numHalls] ** 2)

                (dist_s, dist_f, car_heading) = self.next_car_states(st, flip_sides)

                dist_s2 = self.hallWidths[(st.curHall+1) % self.numHalls] - dist_s
                dist_f2 = dist_f - wall_dist

            if dist_f > LIDAR_RANGE:
                dist_f = LIDAR_RANGE
            if dist_f2 > LIDAR_RANGE:
                dist_f2 = LIDAR_RANGE

            if self.turns[st.curHall] <= 0:
                return np.array([dist_s, dist_s2, dist_f, dist_f2, car_heading])
            else:
                return np.array([dist_s2, dist_s, dist_f2, dist_f, car_heading])

        else:
            return self.scan_lidar(st)

    def is_safe(self, st: State) -> bool:
        corner_angle = np.pi - np.abs(self.turns[st.curHall])
        normal_to_top_wall = [np.sin(corner_angle), -np.cos(corner_angle)]

        # note that dist_f is the x coordinate, and dist_s is the y coordinate
        dot_prod_top = normal_to_top_wall[0] * st.car_dist_f + normal_to_top_wall[1] * st.car_dist_s

        if dot_prod_top <= SAFE_DISTANCE or\
           (dot_prod_top >= (self.hallWidths[(st.curHall+1) % self.numHalls] - SAFE_DISTANCE)
            and st.car_dist_s >= self.hallWidths[(st.curHall) % self.numHalls] - SAFE_DISTANCE) or\
           st.car_dist_s <= SAFE_DISTANCE:
            # print('crash in mode ' + self.name + ' at heading: ' + str(st.car_heading)
            # + ', position: ' + str(st.car_dist_s))
            return False

        return True

    def _reward_fn(self, st0: State, action: np.ndarray, st1: State) -> float:

        delta = action[0]
        reward = STEP_REWARD_GAIN

        corner_dist = np.sqrt((st1.outer_x - st1.inner_x) ** 2 + (st1.outer_y - st1.inner_y) ** 2)
        wall_dist = np.sqrt(corner_dist ** 2 -
                            self.hallWidths[(st1.curHall+1) % self.numHalls] ** 2)

        reward += SPEED_GAIN * st1.car_V
        if st1.car_V < MINIMUM_ACCEPTABLE_SPEED:
            reward += LOW_SPEED_REWARD

        # Region 1
        if st1.car_dist_s > 0 and st1.car_dist_s < self.hallWidths[st1.curHall] and\
           st1.car_dist_f > wall_dist:

            # reward += MOVE_FORWARD_GAIN * (st0.car_dist_f - st1.car_dist_f)

            # only apply these rules if not too close to a turn
            if st1.car_dist_f > LIDAR_RANGE:

                reward += INPUT_REWARD_GAIN * delta * delta
                reward += MIDDLE_REWARD_GAIN * \
                    abs(st1.car_dist_s - self.hallWidths[st1.curHall] / 2.0)

        # Region 2
        elif st1.car_dist_s > 0 and st1.car_dist_s < self.hallWidths[st1.curHall] and\
                st1.car_dist_f <= wall_dist:

            reward += HEADING_GAIN * np.abs(st1.car_heading - self.turns[st1.curHall])
            if not np.sign(st1.car_heading) == np.sign(self.turns[st1.curHall]):
                reward -= 10 * STEP_REWARD_GAIN

        # Region 3
        elif st1.car_dist_s > self.hallWidths[st1.curHall] and\
                st1.car_dist_f <= self.hallWidths[st1.curHall]:

            pass

        # Check for a crash
        is_safe = self.is_safe(st1)
        if not is_safe:
            reward = CRASH_REWARD

        if st1.car_dist_s > self.hallWidths[st1.curHall] and is_safe:

            corner_angle = np.pi - np.abs(self.turns[st1.curHall])

            dist_to_outer_old = np.sqrt(st0.car_dist_s ** 2 + st0.car_dist_f ** 2)
            dist_to_outer_new = np.sqrt(st1.car_dist_s ** 2 + st1.car_dist_f ** 2)
            inner_angle_old = corner_angle - math.atan(st0.car_dist_s / np.abs(st0.car_dist_f))
            inner_angle_new = corner_angle - math.atan(st1.car_dist_s / np.abs(st1.car_dist_f))

            if corner_angle > np.pi/2:
                inner_angle_old = corner_angle - \
                    math.atan(np.abs(st0.car_dist_f) / st0.car_dist_s) - np.pi/2
                inner_angle_new = corner_angle - \
                    math.atan(np.abs(st1.car_dist_f) / st1.car_dist_s) - np.pi/2

            f2_old = np.cos(inner_angle_old) * dist_to_outer_old
            f2_new = np.cos(inner_angle_new) * dist_to_outer_new

            s_new = np.sin(inner_angle_new) * dist_to_outer_new

            reward += MOVE_FORWARD_GAIN * (f2_new - f2_old)

            reward += MIDDLE_REWARD_GAIN * \
                abs(s_new - self.hallWidths[(st1.curHall+1) % self.numHalls] / 2.0)

        return reward

    def next_car_states(self, st: State, flip_sides: bool) -> Tuple[float, float, float]:

        dist_to_outer = np.sqrt(st.car_dist_s ** 2 + st.car_dist_f ** 2)
        corner_angle = np.pi - np.abs(self.turns[st.curHall])
        inner_angle = corner_angle - math.atan(st.car_dist_s / np.abs(st.car_dist_f))

        if corner_angle > np.pi/2:
            inner_angle = corner_angle - math.atan(np.abs(st.car_dist_f) / st.car_dist_s) - np.pi/2

        next_dist_s = np.sin(inner_angle) * dist_to_outer
        if flip_sides:
            next_dist_s = self.hallWidths[(st.curHall+1) % self.numHalls] - next_dist_s

        next_dist_f = self.hallLengths[(st.curHall+1) % self.numHalls] - \
            np.cos(inner_angle) * dist_to_outer

        next_car_heading = st.car_heading - self.turns[st.curHall]
        if next_car_heading > np.pi:
            next_car_heading -= 2 * np.pi
        elif next_car_heading < -np.pi:
            next_car_heading += 2 * np.pi

        return (next_dist_s, next_dist_f, next_car_heading)

    def next_corner_coordinates(self, st: State,
                                flip_sides: bool) -> Tuple[float, float, float, float]:

        # add the length minus the distance from starting outer to inner corner
        if flip_sides:
            starting_corner_dist = np.sqrt((st.outer_x - st.inner_x)
                                           ** 2 + (st.outer_y - st.inner_y) ** 2)
            wall_dist = np.sqrt(starting_corner_dist ** 2 -
                                self.hallWidths[(st.curHall+1) % self.numHalls] ** 2)

            next_outer_x = st.inner_x + \
                np.cos(st.cur_hall_heading) * \
                (self.hallLengths[(st.curHall+1) % self.numHalls] - wall_dist)
            next_outer_y = st.inner_y + \
                np.sin(st.cur_hall_heading) * \
                (self.hallLengths[(st.curHall+1) % self.numHalls] - wall_dist)
        else:
            next_outer_x = st.outer_x + np.cos(st.cur_hall_heading) * \
                self.hallLengths[(st.curHall+1) % self.numHalls]
            next_outer_y = st.outer_y + np.sin(st.cur_hall_heading) * \
                self.hallLengths[(st.curHall+1) % self.numHalls]

        reverse_cur_heading = st.cur_hall_heading - np.pi
        if reverse_cur_heading > np.pi:
            reverse_cur_heading -= 2 * np.pi
        elif reverse_cur_heading < -np.pi:
            reverse_cur_heading += 2 * np.pi

        next_heading = st.cur_hall_heading + self.turns[(st.curHall+1) % self.numHalls]
        if next_heading > np.pi:
            next_heading -= 2 * np.pi
        elif next_heading < -np.pi:
            next_heading += 2 * np.pi

        out_wall_proj_length = np.abs(
            self.hallWidths[st.curHall] / np.sin(self.turns[(st.curHall+1) % self.numHalls]))
        proj_point_x = next_outer_x + np.cos(next_heading) * out_wall_proj_length
        proj_point_y = next_outer_y + np.sin(next_heading) * out_wall_proj_length

        in_wall_proj_length = np.abs(
            self.hallWidths[(st.curHall+1) % self.numHalls] / np.sin(self.turns[(st.curHall+1)
                                                                                % self.numHalls]))
        next_inner_x = proj_point_x + np.cos(reverse_cur_heading) * in_wall_proj_length
        next_inner_y = proj_point_y + np.sin(reverse_cur_heading) * in_wall_proj_length

        return (next_outer_x, next_outer_y, next_inner_x, next_inner_y)

    def scan_lidar(self, st: State) -> np.ndarray:

        car_heading_deg = st.car_heading * 180 / np.pi

        theta_t = np.linspace(-self.lidar_field_of_view,
                              self.lidar_field_of_view, self.lidar_num_rays)

        # lidar measurements
        data = np.zeros(len(theta_t))

        corner_dist = np.sqrt((st.outer_x - st.inner_x) ** 2 + (st.outer_y - st.inner_y) ** 2)
        wall_dist = np.sqrt(corner_dist ** 2 - self.hallWidths[st.curHall] ** 2)

        car_dist_s_inner = self.hallWidths[st.curHall] - st.car_dist_s
        car_dist_f_inner = st.car_dist_f - wall_dist

        region1 = False
        region2 = False
        region3 = False

        if car_dist_f_inner >= 0:

            theta_outer = np.arctan(float(st.car_dist_s) / st.car_dist_f) * 180 / np.pi

            if st.car_dist_s <= self.hallWidths[st.curHall]:

                theta_inner = -np.arctan(float(car_dist_s_inner) / car_dist_f_inner) * 180 / np.pi

                if np.abs(theta_inner) <= np.abs(self.turns[st.curHall]) * 180 / np.pi:
                    region1 = True
                else:
                    region2 = True
            else:

                car_dist_s_inner = st.car_dist_s - self.hallWidths[st.curHall]
                theta_inner = np.arctan(float(car_dist_s_inner) / car_dist_f_inner) * 180 / np.pi
                region3 = True
        else:

            corner_angle = np.pi - np.abs(self.turns[st.curHall])
            normal_to_top_wall = [np.sin(corner_angle), -np.cos(corner_angle)]

            # note that dist_f is the x coordinate, and dist_s is the y coordinate
            dot_prod_top = normal_to_top_wall[0] * \
                st.car_dist_f + normal_to_top_wall[1] * st.car_dist_s

            car_dist_f_inner = np.abs(car_dist_f_inner)

            if car_dist_s_inner >= 0:

                if dot_prod_top >= self.hallWidths[(st.curHall+1) % self.numHalls]:
                    region1 = True
                else:
                    region2 = True

                theta_inner = -90 - np.arctan(float(car_dist_f_inner) /
                                              car_dist_s_inner) * 180 / np.pi

            else:
                car_dist_s_inner = np.abs(car_dist_s_inner)
                theta_inner = 90 + np.arctan(float(car_dist_f_inner) /
                                             car_dist_s_inner) * 180 / np.pi
                region3 = True

            if st.car_dist_f >= 0:
                # car_dist_f_outer = st.car_dist_f
                theta_outer = np.arctan(float(st.car_dist_s) / st.car_dist_f) * 180 / np.pi

            else:
                # car_dist_f_outer = np.abs(st.car_dist_f)
                theta_outer = 90 + np.arctan(np.abs(st.car_dist_f) /
                                             float(st.car_dist_s)) * 180 / np.pi

        car_dist_s_outer = st.car_dist_s

        if self.turns[st.curHall] < 0:
            dist_l = car_dist_s_outer
            dist_r = car_dist_s_inner
            theta_l = theta_outer
            theta_r = theta_inner

        else:
            dist_l = car_dist_s_inner
            dist_r = car_dist_s_outer
            theta_l = -theta_inner
            theta_r = -theta_outer

        # Region 1 (before turn)
        if region1:

            index = 0

            for angle in theta_t:

                angle = angle + car_heading_deg
                if angle > 180:
                    angle = angle - 360
                elif angle < -180:
                    angle = angle + 360

                if angle <= theta_r:
                    data[index] = (dist_r) /\
                        (np.cos((90 + angle) * np.pi / 180))

                elif angle > theta_r and angle <= theta_l:
                    dist_to_outer = np.sqrt(st.car_dist_s ** 2 + st.car_dist_f ** 2)
                    if self.turns[st.curHall] < 0:
                        outer_angle = -self.turns[st.curHall] + theta_l * np.pi / 180
                    else:
                        outer_angle = -self.turns[st.curHall] + theta_r * np.pi / 180
                    dist_to_top_wall = dist_to_outer * np.sin(outer_angle)

                    data[index] = dist_to_top_wall /\
                        np.sin(-self.turns[st.curHall] + angle * np.pi / 180)

                else:

                    data[index] = (dist_l) /\
                        (np.cos((90 - angle) * np.pi / 180))

                # add noise
                data[index] += np.random.uniform(0, self.lidar_noise)

                if data[index] > LIDAR_RANGE or data[index] < 0:
                    data[index] = LIDAR_RANGE

                index += 1

        # Region 2 (during turn)
        elif region2:

            index = 0

            for angle in theta_t:

                angle = angle + car_heading_deg
                if angle > 180:
                    angle = angle - 360
                elif angle < -180:
                    angle = angle + 360

                if self.turns[st.curHall] < 0:
                    if angle <= theta_r:
                        data[index] = (dist_r) /\
                            (np.cos((90 + angle) * np.pi / 180))

                    elif angle > theta_r and angle < self.turns[st.curHall] * 180 / np.pi:
                        dist_to_outer = np.sqrt(st.car_dist_s ** 2 + st.car_dist_f ** 2)
                        outer_angle = -self.turns[st.curHall] + theta_outer * np.pi / 180
                        dist_to_bottom_wall = self.hallWidths[(
                            st.curHall+1) % self.numHalls] - dist_to_outer * np.sin(outer_angle)

                        data[index] = dist_to_bottom_wall /\
                            np.cos(np.pi/2 - self.turns[st.curHall] + angle * np.pi / 180)

                    elif angle > self.turns[st.curHall] * 180 / np.pi and angle <= theta_l:
                        dist_to_outer = np.sqrt(st.car_dist_s ** 2 + st.car_dist_f ** 2)
                        outer_angle = -self.turns[st.curHall] + theta_outer * np.pi / 180
                        dist_to_top_wall = dist_to_outer * np.sin(outer_angle)

                        data[index] = dist_to_top_wall /\
                            np.cos(np.pi/2 + self.turns[st.curHall] - angle * np.pi / 180)
                    else:
                        data[index] = (dist_l) /\
                            (np.cos((90 - angle) * np.pi / 180))

                else:
                    if angle <= theta_r:
                        data[index] = (dist_r) /\
                            (np.cos((90 + angle) * np.pi / 180))

                    elif angle > theta_r and angle <= self.turns[st.curHall] * 180 / np.pi:
                        dist_to_outer = np.sqrt(st.car_dist_s ** 2 + st.car_dist_f ** 2)
                        outer_angle = self.turns[st.curHall] - theta_r * np.pi / 180
                        dist_to_top_wall = dist_to_outer * np.sin(outer_angle)

                        data[index] = dist_to_top_wall /\
                            np.sin(np.pi - self.turns[st.curHall] + angle * np.pi / 180)

                    elif angle > self.turns[st.curHall] * 180 / np.pi and angle <= theta_l:
                        dist_to_outer = np.sqrt(st.car_dist_s ** 2 + st.car_dist_f ** 2)
                        outer_angle = self.turns[st.curHall] - theta_r * np.pi / 180
                        dist_to_bottom_wall = self.hallWidths[(
                            st.curHall+1) % self.numHalls] - dist_to_outer * np.sin(outer_angle)

                        data[index] = dist_to_bottom_wall /\
                            np.cos(np.pi/2 + self.turns[st.curHall] - angle * np.pi / 180)
                    else:
                        data[index] = (dist_l) /\
                            (np.cos((90 - angle) * np.pi / 180))

                # add noise
                data[index] += np.random.uniform(0, self.lidar_noise)

                if data[index] > LIDAR_RANGE or data[index] < 0:
                    data[index] = LIDAR_RANGE

                index += 1

        # Region 3 (after turn)
        elif region3:

            index = 0

            for angle in theta_t:

                angle = angle + car_heading_deg
                if angle > 180:
                    angle = angle - 360
                elif angle < -180:
                    angle = angle + 360

                if self.turns[st.curHall] < 0:
                    if angle < self.turns[st.curHall] * 180 / np.pi:
                        dist_to_outer = np.sqrt(st.car_dist_s ** 2 + st.car_dist_f ** 2)
                        outer_angle = -self.turns[st.curHall] + theta_outer * np.pi / 180
                        dist_to_bottom_wall = self.hallWidths[(
                            st.curHall+1) % self.numHalls] - dist_to_outer * np.sin(outer_angle)

                        data[index] = dist_to_bottom_wall /\
                            np.cos(np.pi/2 - self.turns[st.curHall] + angle * np.pi / 180)

                    elif angle == self.turns[st.curHall] * 180 / np.pi:
                        data[index] = LIDAR_RANGE

                    elif angle >= self.turns[st.curHall] * 180 / np.pi and angle <= theta_l:
                        dist_to_outer = np.sqrt(st.car_dist_s ** 2 + st.car_dist_f ** 2)
                        outer_angle = -self.turns[st.curHall] + theta_outer * np.pi / 180
                        dist_to_top_wall = dist_to_outer * np.sin(outer_angle)

                        data[index] = dist_to_top_wall /\
                            np.cos(np.pi/2 + self.turns[st.curHall] - angle * np.pi / 180)

                    elif angle > theta_l and angle <= theta_r:
                        data[index] = (dist_l) /\
                            (np.cos((90 - angle) * np.pi / 180))

                    else:
                        dist_to_outer = np.sqrt(st.car_dist_s ** 2 + st.car_dist_f ** 2)
                        outer_angle = -self.turns[st.curHall] + theta_outer * np.pi / 180
                        dist_to_bottom_wall = self.hallWidths[(
                            st.curHall+1) % self.numHalls] - dist_to_outer * np.sin(outer_angle)

                        data[index] = dist_to_bottom_wall /\
                            np.cos(np.pi/2 - self.turns[st.curHall] + angle * np.pi / 180)
                else:

                    if angle >= self.turns[st.curHall] * 180 / np.pi:
                        dist_to_outer = np.sqrt(st.car_dist_s ** 2 + st.car_dist_f ** 2)
                        outer_angle = self.turns[st.curHall] - theta_r * np.pi / 180
                        dist_to_bottom_wall = self.hallWidths[(
                            st.curHall+1) % self.numHalls] - dist_to_outer * np.sin(outer_angle)

                        data[index] = dist_to_bottom_wall /\
                            np.cos(np.pi/2 + self.turns[st.curHall] - angle * np.pi / 180)

                    elif angle < self.turns[st.curHall] * 180 / np.pi and angle >= theta_r:
                        dist_to_outer = np.sqrt(st.car_dist_s ** 2 + st.car_dist_f ** 2)
                        outer_angle = self.turns[st.curHall] - theta_r * np.pi / 180
                        dist_to_top_wall = dist_to_outer * np.sin(outer_angle)

                        data[index] = dist_to_top_wall /\
                            np.sin(np.pi - self.turns[st.curHall] + angle * np.pi / 180)

                    elif angle < theta_r and angle >= theta_l:
                        data[index] = (dist_r) /\
                            (np.cos((90 + angle) * np.pi / 180))

                    else:

                        dist_to_outer = np.sqrt(st.car_dist_s ** 2 + st.car_dist_f ** 2)
                        outer_angle = self.turns[st.curHall] - theta_r * np.pi / 180
                        dist_to_bottom_wall = self.hallWidths[(
                            st.curHall+1) % self.numHalls] - dist_to_outer * np.sin(outer_angle)

                        data[index] = dist_to_bottom_wall /\
                            np.cos(np.pi/2 + self.turns[st.curHall] - angle * np.pi / 180)

                # add noise
                data[index] += np.random.uniform(0, self.lidar_noise)

                if data[index] > LIDAR_RANGE or data[index] < 0:
                    data[index] = LIDAR_RANGE

                index += 1

        # add missing rays
        if self.lidar_missing_in_turn_only:

            # add missing rays only in Region 2 (plus an extra 1m before it)
            if st.car_dist_s > 0 and st.car_dist_s < self.hallWidths[st.curHall] and\
               st.car_dist_f <= self.hallWidths[(st.curHall + 1) % self.numHalls] + 1:

                for ray in st.missing_indices:
                    data[ray] = LIDAR_RANGE
        else:
            # add missing rays in all regions
            for ray in st.missing_indices:
                data[ray] = LIDAR_RANGE

        return data

    def render(self, st: State) -> None:
        self.plot_lidar(st)

    def plot_trajectory(self, sts: List[State]) -> None:
        plt.figure()

        self.plotHalls()

        plt.plot([st.car_global_x for st in sts],
                 [st.car_global_y for st in sts],
                 'r--')

        plt.show()

    def plot_lidar(self,
                   st: State,
                   show_halls: bool = True,
                   zero_dist_rays: bool = False,
                   savefilename: str = ''
                   ) -> None:

        plt.figure()

        if show_halls:
            self.plotHalls()

        data = self.scan_lidar(st)

        lidX = []
        lidY = []

        theta_t = np.linspace(-self.lidar_field_of_view,
                              self.lidar_field_of_view, self.lidar_num_rays)

        index = 0

        for curAngle in theta_t:

            if zero_dist_rays and data[index] >= LIDAR_RANGE:
                data[index] = 0

            lidX.append(st.car_global_x + data[index] *
                        np.cos(curAngle * np.pi / 180 + st.car_global_heading))
            lidY.append(st.car_global_y + data[index] *
                        np.sin(curAngle * np.pi / 180 + st.car_global_heading))

            index += 1

        plt.scatter(lidX, lidY, c='green', s=8)

        plt.scatter([st.car_global_x], [st.car_global_y], c='red')

        # plt.ylim((-1,11))
        # plt.xlim((-2, np.max(self.hallLengths) + np.max(self.hallWidths)))

        if len(savefilename) > 0:
            plt.savefig(savefilename)

        plt.show()

    def plot_real_lidar(self, st: State, data, newfig=True):

        if newfig:
            plt.figure()

            self.plotHalls()

        plt.scatter([st.car_global_x], [st.car_global_y], c='red')

        lidX = []
        lidY = []

        theta_t = np.linspace(-self.lidar_field_of_view,
                              self.lidar_field_of_view, self.lidar_num_rays)

        index = 0

        for curAngle in theta_t:

            lidX.append(st.car_global_x + data[index] *
                        np.cos(curAngle * np.pi / 180 + st.car_global_heading))
            lidY.append(st.car_global_y + data[index] *
                        np.sin(curAngle * np.pi / 180 + st.car_global_heading))

            index += 1

        plt.scatter(lidX, lidY, c='green')

        if newfig:
            plt.show()

    def plotHalls(self, ax=plt.gca(), wallwidth=3):

        # 1st hall going up by default and centralized around origin
        # midX = 0
        # midY = 0
        # going_up = True
        # left = True

        prev_outer_x = None
        prev_outer_y = None

        prev_inner_x = None
        prev_inner_y = None
        wall_dist = None

        cur_heading = np.pi / 2

        for i in range(self.numHalls):

            # set up starting and ending outer corners

            # the starting shape of the first hallway will assume a
            # loop (if the hallways do not form a loop, it will just
            # look non-symmetrical)
            if i == 0:
                if self.turns[-1] < 0:
                    l1x1 = -self.hallWidths[0]/2.0
                    l1y1 = -self.hallLengths[0]/2.0

                else:
                    l2x1 = self.hallWidths[0]/2.0
                    l2y1 = -self.hallLengths[0]/2.0

                if self.turns[0] < 0:
                    l1x2 = -self.hallWidths[0]/2.0
                    l1y2 = self.hallLengths[0]/2.0

                else:
                    l2x2 = self.hallWidths[0]/2.0
                    l2y2 = self.hallLengths[0]/2.0
            else:
                if self.turns[i-1] < 0:

                    l1x1 = prev_outer_x
                    l1y1 = prev_outer_y

                    if self.turns[i] < 0:

                        l1x2 = l1x1 + np.cos(cur_heading) * self.hallLengths[i]
                        l1y2 = l1y1 + np.sin(cur_heading) * self.hallLengths[i]

                    # add the length minus the distance from starting outer to inner corner
                    else:

                        l2x2 = prev_inner_x + np.cos(cur_heading) * \
                            (self.hallLengths[i] - wall_dist)
                        l2y2 = prev_inner_y + np.sin(cur_heading) * \
                            (self.hallLengths[i] - wall_dist)

                else:
                    l2x1 = prev_outer_x
                    l2y1 = prev_outer_y

                    # add the length minus the distance from starting outer to inner corner
                    if self.turns[i] < 0:
                        l1x2 = prev_inner_x + np.cos(cur_heading) * \
                            (self.hallLengths[i] - wall_dist)
                        l1y2 = prev_inner_y + np.sin(cur_heading) * \
                            (self.hallLengths[i] - wall_dist)

                    else:

                        l2x2 = l2x1 + np.cos(cur_heading) * self.hallLengths[i]
                        l2y2 = l2y1 + np.sin(cur_heading) * self.hallLengths[i]

            prev_heading = cur_heading - self.turns[i-1]
            reverse_prev_heading = prev_heading - np.pi

            if reverse_prev_heading > np.pi:
                reverse_prev_heading -= 2 * np.pi
            elif reverse_prev_heading < -np.pi:
                reverse_prev_heading += 2 * np.pi

            next_heading = cur_heading + self.turns[i]
            if next_heading > np.pi:
                next_heading -= 2 * np.pi
            elif next_heading < -np.pi:
                next_heading += 2 * np.pi

            reverse_cur_heading = cur_heading - np.pi

            if reverse_cur_heading > np.pi:
                reverse_cur_heading -= 2 * np.pi
            elif reverse_cur_heading < -np.pi:
                reverse_cur_heading += 2 * np.pi

            # rightish turn coming into the current turn (L shape)
            if self.turns[i-1] < 0:

                in_wall_proj_length = np.abs(self.hallWidths[i] / np.sin(self.turns[i-1]))
                proj_point_x = l1x1 + np.cos(reverse_prev_heading) * in_wall_proj_length
                proj_point_y = l1y1 + np.sin(reverse_prev_heading) * in_wall_proj_length

                out_wall_proj_length = np.abs(self.hallWidths[i-1] / np.sin(self.turns[i-1]))
                l2x1 = proj_point_x + np.cos(cur_heading) * out_wall_proj_length
                l2y1 = proj_point_y + np.sin(cur_heading) * out_wall_proj_length

            # _| shape
            else:

                in_wall_proj_length = np.abs(self.hallWidths[i] / np.sin(self.turns[i-1]))
                proj_point_x = l2x1 + np.cos(reverse_prev_heading) * in_wall_proj_length
                proj_point_y = l2y1 + np.sin(reverse_prev_heading) * in_wall_proj_length

                out_wall_proj_length = np.abs(self.hallWidths[i-1] / np.sin(self.turns[i-1]))
                l1x1 = proj_point_x + np.cos(cur_heading) * out_wall_proj_length
                l1y1 = proj_point_y + np.sin(cur_heading) * out_wall_proj_length

            # rightish turn going out of the current turn (Gamma shape)
            next_ind = i+1
            if next_ind >= self.numHalls:
                next_ind = 0
            if self.turns[i] < 0:

                out_wall_proj_length = np.abs(self.hallWidths[i] / np.sin(self.turns[i]))
                proj_point_x = l1x2 + np.cos(next_heading) * out_wall_proj_length
                proj_point_y = l1y2 + np.sin(next_heading) * out_wall_proj_length

                in_wall_proj_length = np.abs(self.hallWidths[next_ind] / np.sin(self.turns[i]))
                l2x2 = proj_point_x + np.cos(reverse_cur_heading) * in_wall_proj_length
                l2y2 = proj_point_y + np.sin(reverse_cur_heading) * in_wall_proj_length

                # update next outer corner
                prev_outer_x = l1x2
                prev_outer_y = l1y2

                prev_inner_x = l2x2
                prev_inner_y = l2y2

            else:

                out_wall_proj_length = np.abs(self.hallWidths[i] / np.sin(self.turns[i]))
                proj_point_x = l2x2 + np.cos(next_heading) * out_wall_proj_length
                proj_point_y = l2y2 + np.sin(next_heading) * out_wall_proj_length

                in_wall_proj_length = np.abs(self.hallWidths[next_ind] / np.sin(self.turns[i]))
                l1x2 = proj_point_x + np.cos(reverse_cur_heading) * in_wall_proj_length
                l1y2 = proj_point_y + np.sin(reverse_cur_heading) * in_wall_proj_length

                # update next outer corner
                prev_outer_x = l2x2
                prev_outer_y = l2y2

                prev_inner_x = l1x2
                prev_inner_y = l1y2

            starting_corner_dist_sq = (l1x2 - l2x2) ** 2 + (l1y2 - l2y2) ** 2
            wall_dist = np.sqrt(starting_corner_dist_sq - self.hallWidths[i] ** 2)

            cur_heading = next_heading

            l1x = np.array([l1x1, l1x2])
            l1y = np.array([l1y1, l1y2])
            l2x = np.array([l2x1, l2x2])
            l2y = np.array([l2y1, l2y2])
            plt.plot(l1x, l1y, 'b', linewidth=wallwidth)
            plt.plot(l2x, l2y, 'b', linewidth=wallwidth)

    def vectorize_state(self, st: State) -> np.ndarray:
        return np.array([st.car_dist_s, st.car_dist_f, st.car_heading, st.car_V])

    def state_from_vector(self, arr: np.ndarray) -> State:
        return self.set_state_local(x=arr[0], y=arr[1], theta=arr[2],
                                    v=arr[3], old_st=self.init_state)


def long_square_hall_right(length=20, width=DEFAULT_HALL_WIDTH):

    short_length = 20
    long_length = length + 20
    hallWidths = [width, width, width, width]
    hallLengths = [long_length, short_length, long_length, short_length]
    turns = [-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2]

    return (hallWidths, hallLengths, turns)


def square_hall_right(side_length=20, width=DEFAULT_HALL_WIDTH):

    hallWidths = [width, width, width, width]
    hallLengths = [side_length, side_length, side_length, side_length]
    turns = [-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2]

    return (hallWidths, hallLengths, turns)


def T_hall_right(width=DEFAULT_HALL_WIDTH):

    hallWidths = [width, width, width, width, width, width, width, width]
    hallLengths = [10, 10, 10, 10, 27, 10, 10, 10]
    turns = [-np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, -np.pi/2]

    return (hallWidths, hallLengths, turns)


def square_hall_left(side_length=20, width=DEFAULT_HALL_WIDTH):

    hallWidths = [width, width, width, width]
    hallLengths = [side_length, side_length, side_length, side_length]
    turns = [np.pi/2, np.pi/2, np.pi/2, np.pi/2]

    return (hallWidths, hallLengths, turns)


def trapezoid_hall_sharp_right(width=DEFAULT_HALL_WIDTH):

    hallWidths = [width, width, width, width]
    hallLengths = [20 + 2 * np.sqrt(200), 20, 20, 20]
    turns = [(-3 * np.pi) / 4, -np.pi/4, -np.pi/4, (-3 * np.pi)/4]

    return (hallWidths, hallLengths, turns)


def trapezoid_hall_sharp_left(width=DEFAULT_HALL_WIDTH):

    hallWidths = [width, width, width, width]
    hallLengths = [20 + 2 * np.sqrt(200), 20, 20, 20]
    turns = [(3 * np.pi) / 4, np.pi/4, np.pi/4, (3 * np.pi)/4]

    return (hallWidths, hallLengths, turns)


def triangle_hall_sharp_right(width=DEFAULT_HALL_WIDTH):

    hallWidths = [width, width, width]
    hallLengths = [30, np.sqrt(1800), 30]
    turns = [(-3 * np.pi) / 4, (-3 * np.pi)/4, -np.pi / 2]

    return (hallWidths, hallLengths, turns)


def triangle_hall_equilateral_right(side_length=20, width=DEFAULT_HALL_WIDTH):

    hallWidths = [width, width, width]
    hallLengths = [side_length, side_length, side_length]
    turns = [(-2 * np.pi) / 3, (-2 * np.pi) / 3, (-2 * np.pi) / 3]

    return (hallWidths, hallLengths, turns)


def triangle_hall_equilateral_left(side_length=20, width=DEFAULT_HALL_WIDTH):

    hallWidths = [width, width, width]
    hallLengths = [side_length, side_length, side_length]
    turns = [(2 * np.pi) / 3, (2 * np.pi) / 3, (2 * np.pi) / 3]

    return (hallWidths, hallLengths, turns)


def trapezoid_hall_slight_right(width=DEFAULT_HALL_WIDTH):

    hallWidths = [width, width, width, width]
    hallLengths = [20, 20, 20 + 2 * np.sqrt(200), 20]
    turns = [-np.pi/4, (-3 * np.pi) / 4,  (-3 * np.pi)/4, -np.pi/4]

    return (hallWidths, hallLengths, turns)


def complex_track(width=DEFAULT_HALL_WIDTH):

    l1 = 20
    l2 = 16
    l3 = 15
    l4 = 15

    y = width / np.sin(np.pi / 3)
    delta = width / np.tan(np.pi / 3)
    z = (l2 - delta) / 2.0
    x = l1 / 2.0 - z - y

    hallWidths = [width, width, width, width, width, width, width, width]
    hallLengths = [l1, l2, l3, l4, 2 * (l3 + x), l4, l3, l2]
    turns = [(-2 * np.pi) / 3, (2 * np.pi) / 3, (-np.pi) / 2, (-np.pi) / 2,
             (-np.pi) / 2, (-np.pi) / 2, (2 * np.pi) / 3, (-2*np.pi) / 3]

    return (hallWidths, hallLengths, turns)


def make_straight(length: float,
                  use_throttle: bool = True,
                  lidar_num_rays: int = 1081,
                  width: float = DEFAULT_HALL_WIDTH,
                  ) -> F110Mode:
    hallWidths, hallLengths, turns = long_square_hall_right(length, width=width)
    return F110Mode(
        name=f'f110_straight_{length}m',
        hallWidths=hallWidths,
        hallLengths=hallLengths,
        turns=turns,
        init_car_dist_s=hallWidths[0]/2.0,
        init_car_dist_f=length + LIDAR_RANGE + 3,
        init_car_heading=0,
        init_car_V=2.4,
        time_step=0.1,
        use_throttle=use_throttle,
        lidar_field_of_view=115,
        lidar_num_rays=lidar_num_rays,
    )


def make_square_right(use_throttle: bool = True,
                      lidar_num_rays: int = 1081,
                      width: float = DEFAULT_HALL_WIDTH,
                      ) -> F110Mode:
    hallWidths, hallLengths, turns = square_hall_right(width=width)
    return F110Mode(
        name='f110_square_right',
        hallWidths=hallWidths,
        hallLengths=hallLengths,
        turns=turns,
        init_car_dist_s=hallWidths[0]/2.0,
        init_car_dist_f=LIDAR_RANGE + 3,
        init_car_heading=0,
        init_car_V=2.4,
        time_step=0.1,
        use_throttle=use_throttle,
        lidar_field_of_view=115,
        lidar_num_rays=lidar_num_rays,
    )


def make_square_left(use_throttle: bool = True,
                     lidar_num_rays: int = 1081,
                     width: float = DEFAULT_HALL_WIDTH,
                     ) -> F110Mode:
    hallWidths, hallLengths, turns = square_hall_left(width=width)
    return F110Mode(
        name='f110_square_left',
        hallWidths=hallWidths,
        hallLengths=hallLengths,
        turns=turns,
        init_car_dist_s=hallWidths[0]/2.0,
        init_car_dist_f=LIDAR_RANGE + 3,
        init_car_heading=0,
        init_car_V=2.4,
        time_step=0.1,
        use_throttle=use_throttle,
        lidar_field_of_view=115,
        lidar_num_rays=lidar_num_rays,
    )


def make_sharp_right(use_throttle: bool = True,
                     lidar_num_rays: int = 1081,
                     width: float = DEFAULT_HALL_WIDTH,
                     ) -> F110Mode:
    hallWidths, hallLengths, turns = triangle_hall_equilateral_right(width=width)
    return F110Mode(
        name='f110_sharp_right',
        hallWidths=hallWidths,
        hallLengths=hallLengths,
        turns=turns,
        init_car_dist_s=hallWidths[0]/2.0,
        init_car_dist_f=LIDAR_RANGE + 3,
        init_car_heading=0,
        init_car_V=2.4,
        time_step=0.1,
        use_throttle=use_throttle,
        lidar_field_of_view=115,
        lidar_num_rays=lidar_num_rays,
    )


def make_sharp_left(use_throttle: bool = True,
                    lidar_num_rays: int = 1081,
                    width: float = DEFAULT_HALL_WIDTH,
                    ) -> F110Mode:
    hallWidths, hallLengths, turns = triangle_hall_equilateral_left(width=width)
    return F110Mode(
        name='f110_sharp_left',
        hallWidths=hallWidths,
        hallLengths=hallLengths,
        turns=turns,
        init_car_dist_s=hallWidths[0]/2.0,
        init_car_dist_f=LIDAR_RANGE + 3,
        init_car_heading=0,
        init_car_V=2.4,
        time_step=0.1,
        use_throttle=use_throttle,
        lidar_field_of_view=115,
        lidar_num_rays=lidar_num_rays,
    )
