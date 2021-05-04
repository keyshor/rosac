import numpy as np
from gym.envs.robotics import rotations, robot_env, utils
import gym.utils
import os
import enum

from hybrid_gym.model import Mode
from typing import List, Dict, Union, NamedTuple, Any

pick_height_offset: float = 0.1
object_length: float = 0.05 # each object is a object_length x object_length x object_length cube
min_obj_dist: float = 0.1 # minimum distance between objects
obj_pos_tolerance: float = 0.1 # norm(obj_pos - point) >= obj_pos_tolerance means that the object is not at the point
height_offset: float = 0.42470206262153437 # height of table's surface
mujoco_xml_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'mujoco_xml'
))
half_object_diagonal: float = 0.5 * np.sqrt(3) * object_length

def goal_distance(goal_a: np.ndarray, goal_b: np.ndarray) -> np.ndarray:
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

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

    def __init__(
        self, model_path: str, n_substeps: int, gripper_extra_height: float,
        block_gripper: bool, num_objects: int,
        target_offset: Union[float, np.ndarray], obj_range: float,
        target_range: float, distance_threshold: float, initial_qpos: Dict,
        reward_type: str, mode_type: ModeType,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            num_objects (int): number of objects in the environment
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
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

        super().__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal: np.ndarray, goal: np.ndarray, info: Any) -> np.ndarray:
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
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
            l = 3 * i
            h = 3 * (i + 1)
            self.sim.model.site_pos[site_id] = self.goal[l:h] - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self) -> bool:
        self.sim.set_state(self.initial_state)

        # randomize start position of objects
        self.num_stack = self.np_random.randint(self.num_objects)
        self.obj_perm = self.np_random.permutation(self.num_objects)
        # object_xpos[i] is the position of object{self.obj_perm[i]}
        object_xpos = [self.initial_gripper_xpos[:2] for i in range(self.num_objects)]
        for i in range(self.num_objects):
            object_name = f'object{self.obj_perm[i]}:joint'
            if 1 <= i < self.num_stack:
                object_xpos[i] = object_xpos[i-1].copy()
            else:
                while np.linalg.norm(object_xpos[i] - self.initial_gripper_xpos[:2]) < 0.1 or \
                        (i >= 1 and np.nanmin([np.linalg.norm(object_xpos[i] - object_xpos[j]) for j in range(i)]) < min_obj_dist):
                    object_xpos[i] = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
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
                self.step([0, 0, 0, -1])

        for _ in range(10):
            self.sim.step()
        self.sim.forward()
        return True

    def _sample_goal(self) -> np.ndarray:
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
        else: # self.mode_type == ModeType.PLACE_OBJ
            desired_arm_position = self.arm_position()
            desired_object_pos[self.obj_perm[self.num_stack]] = desired_object_pos[self.obj_perm[self.num_stack-1]] + [0,0,object_length]
        self.goal_dict['arm'] = desired_arm_position
        for i in range(self.num_objects):
            self.goal_dict[f'obj{i}'] = desired_object_pos[i]
        return np.concatenate(
            [desired_arm_position] + [x.ravel() for x in desired_object_pos]
        )

    def _is_success(self, achieved_goal, desired_goal) -> bool:
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos: Dict[str, np.ndarray]) -> None:
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
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

class State(NamedTuple):
    gripper_pos: np.ndarray
    gripper_fingers: np.ndarray
    object_pos: List[np.ndarray]

class PickPlaceMode(Mode[State]):
    multi_obj: MultiObjectEnv

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
            obj_range=0.15, target_range=0.15, distance_threshold=0.05*num_objects,
            initial_qpos=initial_qpos, reward_type=reward_type, mode_type=mode_type,
        )

    def set_state(self, state: State) -> None:
        self.multi_obj.sim.data.set_mocap_pos('robot0:mocap', state.gripper_pos)
        self.multi_obj.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', state.gripper_fingers[0])
        self.multi_obj.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', state.gripper_fingers[1])
        for i in range(len(state.object_pos)):
            object_name = f'object{i}:joint'
            self.multi_obj.sim.data.set_joint_qpos(object_name, state.object_pos[i])

    def get_state(self) -> State:
        return State(
            gripper_pos = self.multi_obj.arm_position(),
            gripper_fingers = self.multi_obj.gripper_fingers(),
            object_pos = [self.multi_obj.object_qpos(i)
                          for i in range(self.multi_obj.num_objects)],
        )

    def reset(self) -> State:
        self.multi_obj.reset()
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

    def _reward_fn(self, state: State, action: np.ndarray, next_state: State) -> float:
        self.set_state(next_state)
        obs_dict = self.multi_obj._get_obs()
        return self.multi_obj.compute_reward(
            obs_dict['achieved_goal'],
            obs_dict['desired_goal'],
            None,
        )[0]

    def vectorize_state(self, state: State) -> np.ndarray:
        return np.concatenate([state.gripper_pos, state.gripper_fingers] + state.object_pos)

    def state_from_vector(self, vec: np.ndarray) -> State:
        return State(
            gripper_pos=vec[0:3],
            gripper_fingers=vec[3:5],
            object_pos=[vec[3*i+5, 3*i+8] for i in range(self.multi_obj.num_objects)],
        )
