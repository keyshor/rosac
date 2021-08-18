import numpy as np
from hybrid_gym.model import Transition
from hybrid_gym.envs.pick_place.mode import (
    State, ModeType, PickPlaceMode,
    pick_height_offset, object_length
)


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
        multi_obj = self.source_mode.multi_obj
        if multi_obj.in_tower[multi_obj.next_obj_index]:
            return True
        obs_dict = self.source_mode.multi_obj._get_obs()
        return self.source_mode.multi_obj._is_success(
            obs_dict['achieved_goal'], obs_dict['desired_goal']
        )

    def jump(self, target: str, st: State) -> State:
        st_inter = State(
            mujoco_state=st.mujoco_state,
            tower_set=st.tower_set | frozenset([self.source_mode.multi_obj.next_obj_index]),
            tower_pos=st.tower_pos.copy(),
            goal_dict=dict(st.goal_dict),
        )
        self.target_mode.set_state(st)
        self.target_mode.multi_obj.goal = self.target_mode.multi_obj._sample_goal()
        return self.target_mode.get_state()
