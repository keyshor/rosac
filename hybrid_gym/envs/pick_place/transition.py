import numpy as np
from hybrid_gym.model import Transition
from hybrid_gym.envs.pick_place.mode import (
    State, ModeType, PickPlaceMode, pick_height_offset, object_length
)
from typing import Dict, Optional

class PickPlaceTrans(Transition):
    source_mode: PickPlaceMode
    target_mode: PickPlaceMode

    def __init__(self,
                 source_mode: PickPlaceMode,
                 target_mode: PickPlaceMode
                 ) -> None:
        mt = source_mode.multi_obj.mode_type
        next_mt = target_mode.multi_obj.mode_type
        self.source_mode = source_mode
        self.target_mode = target_mode
        super().__init__(str(mt), [str(next_mt)])

    def guard(self, st: State) -> bool:
        self.source_mode.set_state(st)
        obs_dict = self.source_mode.multi_obj._get_obs()
        return self.source_mode.multi_obj._is_success(
            obs_dict['achieved_goal'], obs_dict['desired_goal']
        )

    def jump(self, target: str, st: State) -> State:
        goal_dict = dict(st.goal_dict)
        num_stack = st.num_stack
        target_mode_type = self.target_mode.multi_obj.mode_type
        if target_mode_type == ModeType.MOVE_WITH_OBJ:
            goal_dict['arm'] = goal_dict[f'obj{st.obj_perm[num_stack-1]}'] \
                + np.array([0, 0, pick_height_offset])
            goal_dict[f'obj{st.obj_perm[num_stack]}'] = goal_dict['arm']
        elif target_mode_type == ModeType.MOVE_WITHOUT_OBJ:
            num_stack += 1
            goal_dict['arm'] = goal_dict[f'obj{st.obj_perm[num_stack]}'] \
                + np.array([0, 0, pick_height_offset])
        elif target_mode_type == ModeType.PICK_OBJ:
            goal_dict[f'obj{st.obj_perm[num_stack]}'] = goal_dict['arm']
        else: # target_mode_type == ModeType.PLACE_OBJ
            goal_dict[f'obj{st.obj_perm[num_stack]}'] = \
                goal_dict[f'obj{st.obj_perm[num_stack-1]}'] \
                + [0,0,object_length]
        return State(
            mujoco_state = st.mujoco_state,
            obj_perm = st.obj_perm,
            num_stack = num_stack,
            goal_dict = goal_dict,
        )
