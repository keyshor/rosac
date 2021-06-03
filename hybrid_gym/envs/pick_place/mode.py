import numpy as np
from gym.envs.robotics import rotations, robot_env, utils
import os
import enum
from copy import deepcopy
import mujoco_py

from hybrid_gym.model import Mode
from typing import List, Tuple, Dict, Union, NamedTuple, Any

pick_height_offset: float = 0.1
object_length: float = 0.05  # each object is a object_length x object_length x object_length cube
min_obj_dist: float = 0.1  # minimum distance between objects
# norm(obj_pos - point) >= obj_pos_tolerance means that the object is not at the point
obj_pos_tolerance: float = 0.1
height_offset: float = 0.42470206262153437  # height of table's surface
unsafe_reward: float = -5
mujoco_xml_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'mujoco_xml'))
half_object_diagonal: float = 0.5 * np.sqrt(3) * object_length


def goal_distance(goal_a: np.ndarray, goal_b: np.ndarray) -> float:
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b)


class ModeType(enum.Enum):
    PICK_OBJ = enum.auto()
    PLACE_OBJ = enum.auto()
    MOVE_WITH_OBJ = enum.auto()
    MOVE_WITHOUT_OBJ = enum.auto()


class MultiObjectEnv(robot_env.RobotEnv):
    """
    Superclass for all Stack environments.
    mostly copied from the code for the Fetch environment
    """
    gripper_extra_height: float
    block_gripper: bool
    num_objects: int
    target_offset: Union[float, np.ndarray]
    obj_range: float
    target_range: float
    distance_threshold: float
    reward_type: str
    mode_type: ModeType
    num_stack: int
    obj_perm: np.ndarray
    goal_dict: Dict[str, np.ndarray]
    num_timesteps: int
    max_episode_length: int

    def __init__(
        self, model_path: str, n_substeps: int, gripper_extra_height: float,
        block_gripper: bool, num_objects: int,
        target_offset: Union[float, np.ndarray], obj_range: float,
        target_range: float, distance_threshold: float, initial_qpos: Dict,
        reward_type: str, mode_type: ModeType, max_episode_length: int
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table
                            when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            num_objects (int): number of objects in the environment
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that
                        define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.num_objects = num_objects
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.mode_type = mode_type
        self.num_stack = 0
        self.obj_perm = np.array(range(num_objects))
        self.goal_dict = {}
        self.num_timesteps = 0
        self.max_episode_length = max_episode_length

        super().__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal: np.ndarray, goal: np.ndarray, info: Any) -> float:
        # Compute distance between goal and the achieved goal.
        if not self.is_safe():
            return unsafe_reward
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -float(d > self.distance_threshold)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self) -> None:
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action: np.ndarray) -> None:
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl = action[:3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([action[3], action[3]])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        object_pos = [np.zeros(0) for _ in range(self.num_objects)]
        object_rot = [np.zeros(0) for _ in range(self.num_objects)]
        object_velp = [np.zeros(0) for _ in range(self.num_objects)]
        object_velr = [np.zeros(0) for _ in range(self.num_objects)]
        object_rel_pos = [np.zeros(0) for _ in range(self.num_objects)]
        for i in range(self.num_objects):
            object_name = 'object{}'.format(i)
            object_pos[i] = self.sim.data.get_site_xpos(object_name)
            # rotations
            object_rot[i] = rotations.mat2euler(self.sim.data.get_site_xmat(object_name))
            # velocities
            object_velp[i] = self.sim.data.get_site_xvelp(object_name) * dt
            object_velr[i] = self.sim.data.get_site_xvelr(object_name) * dt
            # gripper state
            object_rel_pos[i] = object_pos[i] - grip_pos
            object_velp[i] -= grip_velp
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        achieved_goal = np.concatenate([self.arm_position()] + [x.ravel() for x in object_pos])
        object_pos_cat = np.concatenate([x.ravel() for x in object_pos])
        object_rot_cat = np.concatenate([x.ravel() for x in object_rot])
        object_velp_cat = np.concatenate([x.ravel() for x in object_velp])
        object_velr_cat = np.concatenate([x.ravel() for x in object_velr])
        object_rel_pos_cat = np.concatenate([x.ravel() for x in object_rel_pos])
        obs = np.concatenate([
            grip_pos, object_pos_cat, object_rel_pos_cat, gripper_state, object_rot_cat,
            object_velp_cat, object_velr_cat, grip_velp, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self) -> None:
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self) -> None:
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        for i in range(self.num_objects):
            site_id = self.sim.model.site_name2id('target{}'.format(i))
            ll = 3 * i
            h = 3 * (i + 1)
            self.sim.model.site_pos[site_id] = self.goal[ll:h] - sites_offset[0]
        self.sim.forward()

    def initialize_positions(self) -> None:
        self.sim.set_state(self.initial_state)
        self.num_timesteps = 0

        # randomize start position of objects
        self.tower_location = self.initial_gripper_xpos[:2] \
            + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
        self.obj_perm = self.np_random.permutation(self.num_objects)
        # object_xpos[i] is the position of object{self.obj_perm[i]}
        object_xpos = [self.initial_gripper_xpos[:2] for i in range(self.num_objects)]
        for i in range(self.num_objects):
            object_name = f'object{self.obj_perm[i]}:joint'
            if i < self.num_stack:
                object_xpos[i] = self.tower_location.copy()
            else:
                while np.linalg.norm(object_xpos[i] - self.initial_gripper_xpos[:2]) < 0.1 or \
                        np.linalg.norm(object_xpos[i] - self.tower_location) < 0.1 or \
                        (i >= 1 and np.nanmin([np.linalg.norm(object_xpos[i] - object_xpos[j])
                                               for j in range(i)]) < min_obj_dist):
                    object_xpos[i] = self.initial_gripper_xpos[:2] + \
                        self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos(object_name)
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos[i]
            if i < self.num_stack:
                object_qpos[2] = self.height_offset + i * object_length
            else:
                object_qpos[2] = self.height_offset
            self.sim.data.set_joint_qpos(object_name, object_qpos)

        # set gripper location
        if self.mode_type == ModeType.PICK_OBJ \
                or self.mode_type == ModeType.MOVE_WITH_OBJ:
            gripper_target = self.object_position(self.obj_perm[self.num_stack]) \
                + np.array([0, 0, pick_height_offset])
        else:
            gripper_target = self.object_position(self.obj_perm[self.num_stack-1]) \
                + np.array([0, 0, pick_height_offset])
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)

        for _ in range(10):
            self.sim.step()
        self.sim.forward()

        # place object in gripper's grip
        self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', half_object_diagonal)
        self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', half_object_diagonal)
        if self.mode_type == ModeType.PLACE_OBJ \
                or self.mode_type == ModeType.MOVE_WITH_OBJ:
            object_index = self.obj_perm[self.num_stack]
            object_name = 'object{}:joint'.format(object_index)
            for _ in range(3):
                object_qpos = self.object_qpos(object_index)
                object_qpos[0:3] = self.arm_position()
                self.sim.data.set_joint_qpos(object_name, object_qpos)
                self.step(np.array([0, 0, 0, -1]))

        for _ in range(5):
            self.sim.step()
        self.sim.forward()

    def _reset_sim(self) -> bool:
        self.num_stack = self.np_random.randint(self.num_objects)
        self.initialize_positions()
        return True

    def make_goal_dict(self) -> Dict[str, np.ndarray]:
        desired_object_pos = [self.object_position(i) for i in range(self.num_objects)]
        if self.mode_type == ModeType.MOVE_WITH_OBJ:
            desired_arm_position = self.object_position(self.obj_perm[self.num_stack-1]) \
                + np.array([0, 0, pick_height_offset])
            desired_object_pos[self.obj_perm[self.num_stack]] = desired_arm_position
        elif self.mode_type == ModeType.MOVE_WITHOUT_OBJ:
            desired_arm_position = self.object_position(self.obj_perm[self.num_stack]) \
                + np.array([0, 0, pick_height_offset])
        elif self.mode_type == ModeType.PICK_OBJ:
            desired_arm_position = self.arm_position()
            desired_object_pos[self.obj_perm[self.num_stack]] = desired_arm_position
        else:  # self.mode_type == ModeType.PLACE_OBJ
            desired_arm_position = self.arm_position()
            desired_object_pos[self.obj_perm[self.num_stack]
                               ] = desired_object_pos[self.obj_perm[self.num_stack-1]] \
                + [0, 0, object_length]
        return dict(
            [('arm', desired_arm_position)] +
            [(f'obj{i}', desired_object_pos[i])
             for i in range(self.num_objects)]
        )

    def goal_vector(self, goal_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [goal_dict['arm']] +
            [goal_dict[f'obj{i}'].ravel()
             for i in range(self.num_objects)]
        )

    def _sample_goal(self) -> np.ndarray:
        self.goal_dict = self.make_goal_dict()
        return self.goal_vector(self.goal_dict)

    def _is_success(self, achieved_goal, desired_goal) -> bool:
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold)

    def _env_setup(self, initial_qpos: Dict[str, np.ndarray]) -> None:
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]
                                  ) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        self.height_offset = height_offset

    def render(self, mode: str = 'human', width: int = 500, height: int = 500) -> None:
        return super().render(mode, width, height)

    # override step()
    def step(self, action: np.ndarray) -> Tuple[
            Dict[str, np.ndarray],
            float,
            bool,
            Dict[str, Any]
    ]:
        self.num_timesteps += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        is_success = self._is_success(obs['achieved_goal'], self.goal)
        done = is_success or not self.is_safe() or self.num_timesteps > self.max_episode_length
        info = {'is_success': is_success}
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def object_qpos(self, object_index: int) -> np.ndarray:
        if object_index < 0 or object_index >= self.num_objects:
            raise KeyError('valid object indices i satisfy 0 <= i < {}'.format(self.num_objects))
        object_name = 'object{}:joint'.format(object_index)
        object_qpos = self.sim.data.get_joint_qpos(object_name)
        return object_qpos

    def object_position(self, object_index: int) -> np.ndarray:
        return self.object_qpos(object_index)[0:3]

    def arm_position(self) -> np.ndarray:
        return self.sim.data.get_site_xpos('robot0:grip').copy()

    def gripper_fingers(self) -> np.ndarray:
        gripper_l = self.sim.data.get_joint_qpos('robot0:l_gripper_finger_joint')
        gripper_r = self.sim.data.get_joint_qpos('robot0:r_gripper_finger_joint')
        return np.array([gripper_l, gripper_r])

    def on_top_of(self, a_index: int, b_index: int) -> bool:
        a_xpos = self.object_position(a_index)
        b_xpos = self.object_position(b_index)
        return np.linalg.norm(a_xpos[0:2] - b_xpos[0:2]) < 0.5 * object_length \
            and np.fabs(a_xpos[2] - b_xpos[2] - object_length) < 0.015

    def is_grasp(self, object_index: int) -> bool:
        object_xpos = self.object_position(object_index)
        gripper_xpos = self.arm_position()
        gripper_fingers = self.gripper_fingers()
        gripper_l = gripper_fingers[0]
        gripper_r = gripper_fingers[1]
        return np.linalg.norm(object_xpos - gripper_xpos) < 0.5 * object_length \
            and gripper_l + gripper_r < 1.01 * object_length


    def is_safe(self) -> bool:
        for i in range(self.num_stack):
            if np.linalg.norm(self.object_position(self.obj_perm[i]) -
                              self.goal_dict[f'obj{self.obj_perm[i]}']) \
                    >= obj_pos_tolerance:
                return False
        return True

class State(NamedTuple):
    mujoco_state: mujoco_py.MjSimState
    obj_perm: np.ndarray
    num_stack: int
    goal_dict: Dict[str, np.ndarray]


class PickPlaceMode(Mode[State]):
    multi_obj: MultiObjectEnv
    obj_perm_index: int
    num_stack_index: int
    goal_arm_index: int
    goal_obj_index: int

    def __init__(self, mode_type: ModeType, num_objects=3, reward_type='sparse'):
        model_xml_path = os.path.join(
            mujoco_xml_path, f'object{num_objects}.xml'
        )
        initial_qpos: Dict[str, Union[float, List[float]]] = {
            'object{}:joint'.format(i): [1.25, 0.53, 0.4, 1., 0., 0., 0.]
            for i in range(num_objects)
        }
        initial_qpos.update({
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        })
        self.multi_obj = MultiObjectEnv(
            model_path=model_xml_path, num_objects=num_objects, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.02*np.sqrt(num_objects),
            initial_qpos=initial_qpos, reward_type=reward_type, mode_type=mode_type,
            max_episode_length=50,
        )
        self.obj_perm_index = -4 * num_objects - 4
        self.num_stack_index = -3 * num_objects - 4
        # goal_arm_index = -3 * num_objects - 3
        # goal_obj_index = -3 * num_objects
        super().__init__(
            name=str(mode_type),
            action_space=self.multi_obj.action_space,
            observation_space=self.multi_obj.observation_space['observation'],
            goal_space=self.multi_obj.observation_space['desired_goal'],
        )

    def set_state(self, state: State) -> None:
        #self.multi_obj.sim.reset()
        #old_mujoco_state = self.multi_obj.sim.get_state()
        self.multi_obj.sim.set_state(mujoco_py.MjSimState(
            #old_mujoco_state.time,
            state.mujoco_state.time,
            state.mujoco_state.qpos,
            state.mujoco_state.qvel,
            state.mujoco_state.act,
            state.mujoco_state.udd_state
        ))
        self.multi_obj.obj_perm = state.obj_perm.copy()
        self.multi_obj.num_stack = state.num_stack
        self.multi_obj.goal_dict = dict(state.goal_dict)
        self.multi_obj.goal = self.multi_obj.goal_vector(state.goal_dict)

    def get_state(self) -> State:
        return State(
            mujoco_state=deepcopy(self.multi_obj.sim.get_state()),
            obj_perm=self.multi_obj.obj_perm.copy(),
            num_stack=self.multi_obj.num_stack,
            goal_dict=dict(self.multi_obj.goal_dict),
        )

    def reset(self) -> State:
        self.multi_obj.reset()
        return self.get_state()

    def end_to_end_reset(self) -> State:
        assert self.multi_obj.mode_type == ModeType.MOVE_WITHOUT_OBJ
        self.multi_obj.num_stack = 0
        self.multi_obj.initialize_positions()
        self.multi_obj._sample_goal()
        return self.get_state()

    def is_safe(self, state: State) -> bool:
        for i in range(self.multi_obj.num_stack):
            if np.linalg.norm(self.multi_obj.object_position(self.multi_obj.obj_perm[i]) -
                              self.multi_obj.goal_dict[f'obj{self.multi_obj.obj_perm[i]}']) \
                    >= obj_pos_tolerance:
                return False
        return True

    def render(self, state: State) -> None:
        self.set_state(state)
        self.multi_obj.render()

    def _step_fn(self, state: State, action: np.ndarray) -> State:
        self.set_state(state)
        self.multi_obj.step(action)
        return self.get_state()

    def _observation_fn(self, state: State) -> np.ndarray:
        self.set_state(state)
        return self.multi_obj._get_obs()['observation']

    def achieved_goal(self, state: State) -> np.ndarray:
        self.set_state(state)
        return self.multi_obj._get_obs()['achieved_goal']

    def desired_goal(self, state: State) -> np.ndarray:
        self.set_state(state)
        return self.multi_obj._get_obs()['desired_goal']

    def _reward_fn(self, state: State, action: np.ndarray, next_state: State) -> float:
        self.set_state(next_state)
        obs_dict = self.multi_obj._get_obs()
        return self.multi_obj.compute_reward(
            obs_dict['achieved_goal'],
            obs_dict['desired_goal'],
            None,
        )

    def vectorize_state(self, state: State) -> np.ndarray:
        flattened_mujoco_state = state.mujoco_state.flatten()
        flattened_goal_dict = self.multi_obj.goal_vector(state.goal_dict)
        return np.concatenate([
            flattened_mujoco_state,
            state.obj_perm,
            [state.num_stack],
            flattened_goal_dict,
        ])

    def state_from_vector(self, vec: np.ndarray) -> State:
        return State(
            mujoco_state=mujoco_py.MjSimState.from_flattened(
                vec[0: self.obj_perm_index],
                self.multi_obj.sim,
            ),
            obj_perm=vec[self.obj_perm_index: self.num_stack_index],
            num_stack=vec[self.num_stack_index],
            goal_dict=dict(
                [('arm', vec[self.goal_arm_index: self.goal_obj_index])] +
                [(f'obj{i}', vec[self.goal_obj_index + 3*i: self.goal_obj_index + 3*i + 3])
                 for i in range(self.multi_obj.num_objects)]
            ),
        )
