import numpy as np
import os
import enum
import mujoco_py

from copy import deepcopy
from hybrid_gym.model import Mode
import gym
from gym.envs.robotics import rotations, robot_env, utils
from typing import List, Tuple, Dict, FrozenSet, Union, Optional, NamedTuple, Any

pick_height_offset: float = 0.05
object_length: float = 0.05  # each object is a object_length x object_length x object_length cube
min_obj_dist: float = 0.1  # minimum distance between objects
# norm(obj_pos - point) >= obj_pos_tolerance means that the object is not at the point
obj_pos_tolerance: float = 0.02
finger_pos_scale: float = 1.0
height_offset: float = 0.42470206262153437  # height of table's surface
unsafe_reward: float = -50
mujoco_xml_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'mujoco_xml'))


def goal_distance(goal_a: np.ndarray, goal_b: np.ndarray) -> float:
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b)


class ModeType(enum.Enum):
    PICK_OBJ_PT1 = enum.auto()
    PICK_OBJ_PT2 = enum.auto()
    PICK_OBJ_PT3 = enum.auto()
    PLACE_OBJ_PT1 = enum.auto()
    PLACE_OBJ_PT2 = enum.auto()
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
    fixed_tower_height: Optional[int]
    target_offset: Union[float, np.ndarray]
    obj_range: float
    target_range: float
    distance_threshold: float
    reward_type: str
    mode_type: ModeType
    next_obj_index: int
    tower_pos: np.ndarray
    goal_dict: Dict[str, np.ndarray]
    tower_set: FrozenSet[int]

    def __init__(
        self, model_path: str, n_substeps: int, gripper_extra_height: float,
        block_gripper: bool, num_objects: int, fixed_tower_height: Optional[int],
        target_offset: Union[float, np.ndarray], obj_range: float,
        target_range: float, distance_threshold: float, initial_qpos: Dict,
        reward_type: str, mode_type: ModeType, next_obj_index: int,
    ) -> None:
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
        self.fixed_tower_height = fixed_tower_height
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.mode_type = mode_type
        self.next_obj_index = next_obj_index
        self.tower_pos = np.zeros(shape=(2,))
        self.goal_dict = {}
        self.tower_set = frozenset()

        super().__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal: np.ndarray, goal: np.ndarray, info: Any) -> float:
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return float(self._is_success(achieved_goal, goal))
        else:
            reward = -d + obj_pos_tolerance
            if not self.is_safe():
               reward += unsafe_reward
            if self._is_success(achieved_goal, goal):
                reward += 1e1
            return reward

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

        achieved_goal = np.concatenate([
            self.arm_position(),
            finger_pos_scale * self.gripper_fingers(),
            self.object_position(self.next_obj_index),
        ])
        #object_pos_cat = np.concatenate([x.ravel() for x in object_pos])
        #object_rot_cat = np.concatenate([x.ravel() for x in object_rot])
        #object_velp_cat = np.concatenate([x.ravel() for x in object_velp])
        #object_velr_cat = np.concatenate([x.ravel() for x in object_velr])
        #object_rel_pos_cat = np.concatenate([x.ravel() for x in object_rel_pos])
        #obs = np.concatenate([
        #    grip_pos, object_pos_cat, object_rel_pos_cat, gripper_state, object_rot_cat,
        #    object_velp_cat, object_velr_cat, grip_velp, gripper_vel,
        #])
        obs = np.zeros(0)

        return {
            'observation': np.array(obs.copy(), dtype=np.float32),
            'achieved_goal': np.array(achieved_goal.copy(), dtype=np.float32),
            'desired_goal': np.array(self.goal.copy(), dtype=np.float32),
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

    def initialize_positions(self, tower_height: int) -> None:
        self.sim.set_state(self.initial_state)
        other_objects = np.array(
            [i for i in range(self.num_objects) if i != self.next_obj_index],
            dtype=int,
        )

        # randomize start position of objects
        self.tower_pos = self.initial_gripper_xpos[:2] \
            + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
        obj_perm = self.np_random.permutation(other_objects)
        tower_blocks = obj_perm[:tower_height]
        non_tower_blocks = np.array([self.next_obj_index] + list(obj_perm[tower_height:]))
        self.tower_set = frozenset(tower_blocks)
        object_xpos = np.empty(shape=(self.num_objects,3))
        object_xpos[:,0:2] = self.initial_gripper_xpos[:2]
        object_xpos[:,2] = self.height_offset
        for i in range(tower_height):
            obj_index = tower_blocks[i]
            object_name = f'object{obj_index}:joint'
            object_xpos[obj_index,0:2] = self.tower_pos \
                + self.np_random.uniform(-0.1 * object_length, 0.1 * object_length, size=2)
            object_xpos[obj_index,2] += i * object_length
        for i in range(self.num_objects - tower_height):
            obj_index = non_tower_blocks[i]
            object_name = f'object{obj_index}:joint'
            while np.linalg.norm(object_xpos[obj_index,0:2] - self.initial_gripper_xpos[:2]) < min_obj_dist or \
                    np.linalg.norm(object_xpos[obj_index,0:2] - self.tower_pos) < min_obj_dist or \
                    np.any(np.linalg.norm(
                        object_xpos[obj_index,0:2] - object_xpos[non_tower_blocks[:i],0:2], axis=1,
                    ) < min_obj_dist):
                object_xpos[obj_index,0:2] = self.initial_gripper_xpos[:2] + \
                    self.np_random.uniform(-self.obj_range, self.obj_range, size=2)

        # set gripper location
        if self.mode_type == ModeType.PICK_OBJ_PT1:
            gripper_target = object_xpos[self.next_obj_index] \
                + np.array([0, 0, pick_height_offset])
        elif self.mode_type == ModeType.PICK_OBJ_PT2 \
                or self.mode_type == ModeType.PICK_OBJ_PT3:
            gripper_target = object_xpos[self.next_obj_index]
        elif self.mode_type == ModeType.MOVE_WITH_OBJ:
            gripper_target = object_xpos[self.next_obj_index] \
                + np.array([0, 0, pick_height_offset])
        elif self.mode_type == ModeType.PLACE_OBJ_PT1 \
                or self.mode_type == ModeType.MOVE_WITHOUT_OBJ:
            gripper_target = self.top_tower_block_pos() \
                + np.array([0, 0, object_length + pick_height_offset])
        else:  # self.modeType == ModeType.PLACE_OBJ_PT2
            gripper_target = self.top_tower_block_pos() \
                + np.array([0, 0, object_length])
        gripper_target[:2] += self.np_random.uniform(
            low=-2*self.distance_threshold,
            high=2*self.distance_threshold,
            size=2,
        )
        self.set_gripper_position(gripper_target)

        for obj_index in range(self.num_objects):
            self.set_object_position(obj_index, object_xpos[obj_index])

        for _ in range(5):
            self.sim.step()
        self.sim.forward()

        # place object in gripper's grip
        self.set_gripper_fingers(np.full(shape=(2,), fill_value=object_length))
        if self.mode_type == ModeType.PLACE_OBJ_PT1 \
                or self.mode_type == ModeType.PLACE_OBJ_PT2 \
                or self.mode_type == ModeType.MOVE_WITH_OBJ \
                or self.mode_type == ModeType.PICK_OBJ_PT2 \
                or self.mode_type == ModeType.PICK_OBJ_PT3:
            object_index = self.next_obj_index
            object_name = 'object{}:joint'.format(object_index)
            for _ in range(3):
                object_qpos = self.object_qpos(object_index)
                object_qpos[0:3] = self.arm_position()
                self.sim.data.set_joint_qpos(object_name, object_qpos)
                self.step(np.array([0, 0, 0, -1]))

        # for PICK_OBJ_PT2, the gripper fingers need to be apart
        if self.mode_type == ModeType.PICK_OBJ_PT2:
            for _ in range(3):
                self.step(np.array([0, 0, 0, 1]))
            self.set_gripper_fingers(np.full(shape=(2,), fill_value=object_length))

        for _ in range(5):
            self.sim.step()
        self.sim.forward()

    def set_gripper_position(self, raw_gripper_target: np.ndarray) -> None:
        gripper_target: np.ndarray = raw_gripper_target + np.array([-0.0001, 0.0000, 0.0203])
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()
        self.sim.forward()

    def set_gripper_fingers(self, fingers: np.ndarray) -> None:
        self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', fingers[0])
        self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', fingers[1])

    def set_object_position(self, obj_index: int, xpos: np.ndarray) -> None:
        object_name = f'object{obj_index}:joint'
        object_qpos = self.sim.data.get_joint_qpos(object_name)
        assert object_qpos.shape == (7,)
        object_qpos[:3] = xpos
        self.sim.data.set_joint_qpos(object_name, object_qpos)

    def _reset_sim(self) -> bool:
        tower_height = self.np_random.randint(self.num_objects) \
                if self.fixed_tower_height is None else self.fixed_tower_height
        self.initialize_positions(tower_height=tower_height)
        return True

    def make_goal_dict(self) -> Dict[str, np.ndarray]:
        desired_object_pos = [self.object_position(i) for i in range(self.num_objects)]
        if self.mode_type == ModeType.MOVE_WITH_OBJ:
            desired_arm_position: np.ndarray = self.top_tower_block_pos() \
                + np.array([0, 0, object_length + pick_height_offset])
            desired_finger_pos = np.full((2,), object_length / 2.0)
            desired_object_pos[self.next_obj_index] = desired_arm_position
        elif self.mode_type == ModeType.MOVE_WITHOUT_OBJ:
            desired_arm_position = self.object_position(self.next_obj_index) \
                + np.array([0, 0, pick_height_offset])
            desired_finger_pos = np.full((2,), object_length)
        elif self.mode_type == ModeType.PICK_OBJ_PT1:
            desired_arm_position = self.object_position(self.next_obj_index)
            desired_finger_pos = np.full((2,), object_length)
        elif self.mode_type == ModeType.PICK_OBJ_PT2:
            desired_arm_position = self.object_position(self.next_obj_index)
            desired_finger_pos = np.full((2,), object_length / 2.0)
        elif self.mode_type == ModeType.PICK_OBJ_PT3:
            desired_arm_position = self.arm_position() \
                + np.array([0, 0, pick_height_offset])
            desired_finger_pos = np.full((2,), object_length / 2.0)
            desired_object_pos[self.next_obj_index] = desired_arm_position
        elif self.mode_type == ModeType.PLACE_OBJ_PT1:
            desired_object_pos[self.next_obj_index] = self.top_tower_block_pos() \
                + np.array([0, 0, object_length])
            desired_arm_position = desired_object_pos[self.next_obj_index].copy()
            desired_finger_pos = np.full((2,), object_length / 2.0)
        else:  # self.mode_type == ModeType.PLACE_OBJ_PT2
            desired_arm_position = desired_object_pos[self.next_obj_index] \
                + np.array([0, 0, pick_height_offset])
            desired_finger_pos = np.full((2,), object_length)
        return dict(
            [('arm', desired_arm_position), ('finger', desired_finger_pos)] +
            [(f'obj{i}', desired_object_pos[i])
             for i in range(self.num_objects)]
        )

    def goal_vector(self, goal_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([
            goal_dict['arm'],
            finger_pos_scale * goal_dict['finger'],
            goal_dict[f'obj{self.next_obj_index}'],
        ])

    #def unvectorize_goal(self, vec: np.ndarray) -> Dict[str, np.ndarray]:
    #    return dict(
    #        [('arm', vec[0:3]), ('finger', vec[3:5])] +
    #        [(f'obj{i}', vec[5 + 3*i : 8 + 3*i])
    #         for i in range(self.num_objects)],
    #    )

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
        action_any: Any = action
        action_clip = np.clip(action_any, self.action_space.low, self.action_space.high)
        self._set_action(action_clip)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        is_success = self._is_success(obs['achieved_goal'], self.goal)
        done = is_success or not self.is_safe()
        info = {'is_success': is_success}
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def top_tower_block_pos(self) -> np.ndarray:
        tower_height = len(self.tower_set)
        pos = np.empty(shape=(3,))
        pos[0:2] = self.tower_pos
        pos[2] = self.height_offset + tower_height * object_length
        return pos

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
        for i in range(self.num_objects):
            if i in self.tower_set \
                    and np.linalg.norm(self.object_position(i) -
                                       self.goal_dict[f'obj{i}']) \
                    >= obj_pos_tolerance:
                return False
        return True


class State(NamedTuple):
    mujoco_state: mujoco_py.MjSimState
    tower_set: FrozenSet[int]
    tower_pos: np.ndarray
    goal_dict: Dict[str, np.ndarray]


class PickPlaceMode(Mode[State]):
    multi_obj: MultiObjectEnv
    multi_obj_kwargs: Dict[str, Any]
    fixed_tower_height: Optional[int]
    flatten_obs: bool

    def __init__(self,
                 mode_type: ModeType,
                 next_obj_index: int,
                 num_objects: int = 3,
                 fixed_tower_height: Optional[int] = None,
                 reward_type: str = 'sparse',
                 distance_threshold: float = 0.005,
                 flatten_obs: bool = False,
                 ) -> None:
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
        self.multi_obj_kwargs = dict(
            model_path=model_xml_path, num_objects=num_objects,
            fixed_tower_height=fixed_tower_height,
            block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=distance_threshold,
            initial_qpos=initial_qpos, reward_type=reward_type, mode_type=mode_type,
            next_obj_index=next_obj_index,
        )
        tmp_multi_obj = MultiObjectEnv(**self.multi_obj_kwargs)
        self.fixed_tower_height = fixed_tower_height
        self.flatten_obs = flatten_obs
        obs_space = gym.spaces.utils.flatten_space(
            tmp_multi_obj.observation_space
        ) if flatten_obs else tmp_multi_obj.observation_space
        super().__init__(
            name=f'{mode_type.name}_{next_obj_index}' \
                    if fixed_tower_height is None \
                    else f'{mode_type.name}_h{fixed_tower_height}_{next_obj_index}',
            action_space=tmp_multi_obj.action_space,
            observation_space=gym.spaces.utils.flatten_space(
                tmp_multi_obj.observation_space
            ) if flatten_obs else tmp_multi_obj.observation_space,
        )
        del tmp_multi_obj

    def force_multi_obj(self) -> None:
        if not hasattr(self, 'multi_obj'):
            self.multi_obj = MultiObjectEnv(**self.multi_obj_kwargs)

    def set_state(self, state: State) -> None:
        self.force_multi_obj()
        self.multi_obj.sim.set_state(mujoco_py.MjSimState(
            state.mujoco_state.time,
            state.mujoco_state.qpos,
            state.mujoco_state.qvel,
            state.mujoco_state.act,
            state.mujoco_state.udd_state
        ))
        self.multi_obj.tower_set = state.tower_set
        self.multi_obj.tower_pos = state.tower_pos.copy()
        self.multi_obj.goal_dict = dict(state.goal_dict)
        self.multi_obj.goal = self.multi_obj.goal_vector(state.goal_dict)
        self.multi_obj.sim.forward()

    def get_state(self) -> State:
        self.force_multi_obj()
        return State(
            mujoco_state=deepcopy(self.multi_obj.sim.get_state()),
            tower_set=self.multi_obj.tower_set,
            tower_pos=self.multi_obj.tower_pos.copy(),
            goal_dict=dict(self.multi_obj.goal_dict),
        )

    def reset(self) -> State:
        self.force_multi_obj()
        self.multi_obj.reset()
        return self.get_state()

    def end_to_end_reset(self) -> State:
        self.force_multi_obj()
        if self.multi_obj.mode_type != ModeType.MOVE_WITHOUT_OBJ:
            # print warning?
            pass
        self.multi_obj.initialize_positions(tower_height=0)
        self.multi_obj._sample_goal()
        return self.get_state()

    def is_safe(self, state: State) -> bool:
        self.set_state(state)
        return self.multi_obj.is_safe()

    def is_success(self, state: State) -> bool:
        self.set_state(state)
        obs = self.multi_obj._get_obs()
        return self.multi_obj._is_success(obs['achieved_goal'], obs['desired_goal'])

    def render(self, state: State) -> None:
        self.set_state(state)
        self.multi_obj.render()

    def _step_fn(self, state: State, action: np.ndarray) -> State:
        self.set_state(state)
        self.multi_obj.step(action)
        return self.get_state()

    def compute_reward(self,
                       achieved_goal: np.ndarray,
                       desired_goal: np.ndarray,
                       info: Any,
                       ) -> float:
        return self.multi_obj.compute_reward(
            achieved_goal,
            desired_goal,
            None,
        )

    def _observation_fn(self, state: State) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        self.set_state(state)
        dict_obs = self.multi_obj._get_obs()
        return gym.spaces.utils.flatten(
            self.multi_obj.observation_space, dict_obs
        ) if self.flatten_obs else dict_obs

    def _reward_fn(self, state: State, action: np.ndarray, next_state: State) -> float:
        self.set_state(next_state)
        obs_dict = self.multi_obj._get_obs()
        return self.compute_reward(
            obs_dict['achieved_goal'],
            obs_dict['desired_goal'],
            None,
        )

    def vectorize_state(self, state: State) -> np.ndarray:
        self.set_state(state)
        return np.concatenate([
            [len(self.multi_obj.tower_set)],
            self.multi_obj.tower_pos.copy(),
            self.multi_obj.arm_position(),
            self.multi_obj.gripper_fingers(),
        ] + [
            self.multi_obj.object_position(i)
            for i in range(self.multi_obj.num_objects)
        ])

    def state_from_vector(self, vec: np.ndarray) -> State:
        self.force_multi_obj()
        tower_height = int(vec[0])
        tower_pos = vec[1:3]
        arm_position = vec[3:6]
        gripper_fingers = vec[6:8]
        object_position = np.reshape(
            vec[8:],
            newshape=(self.multi_obj.num_objects,3),
            order='C',
        )

        # compute which objects should be in the tower
        tower_distances = np.linalg.norm(object_position[:,0:2] - tower_pos, axis=1)
        tower_set = frozenset(np.argsort(tower_distances)[:tower_height])
        tower_order = np.argsort(object_position[:,2])
        j = 0
        for i in range(self.multi_obj.num_objects):
            obj_index = tower_order[i]
            if obj_index not in tower_set:
                continue
            object_position[obj_index,2] = self.multi_obj.height_offset + j * object_length
            j += 1
            if np.any(np.abs(object_position[obj_index,0:2] - tower_pos) > 0.1 * object_length):
                object_position[obj_index,0:2] = tower_pos \
                    + self.multi_obj.np_random.uniform(
                        -0.1 * object_length, 0.1 * object_length, size=2
                    )

        self.multi_obj.set_gripper_position(arm_position)
        self.multi_obj.set_gripper_fingers(gripper_fingers)
        for i in range(self.multi_obj.num_objects):
            self.multi_obj.set_object_position(i, object_position[i])
        for _ in range(1):
            self.multi_obj.sim.step()
        self.multi_obj.sim.forward()
        self.multi_obj.tower_pos = tower_pos
        self.multi_obj.tower_set = tower_set
        self.multi_obj.goal = self.multi_obj._sample_goal()
        return self.get_state()
