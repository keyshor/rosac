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
        self.source_mode = source_mode
        self.target_mode = target_mode
        super().__init__(source_mode.name, [target_mode.name])

    def guard(self, st: State) -> bool:
        self.target_mode.force_multi_obj()
        self.source_mode.set_state(st)
        multi_obj = self.source_mode.multi_obj
        if self.target_mode.fixed_tower_height is not None \
                and len(multi_obj.tower_set) != self.target_mode.fixed_tower_height:
            return False
        if multi_obj.next_obj_index in multi_obj.tower_set:
            return True
        obs_dict = self.source_mode.multi_obj._get_obs()
        return self.source_mode.multi_obj._is_success(
            obs_dict['achieved_goal'], obs_dict['desired_goal']
        )

    def jump(self, target: str, st: State) -> State:
        self.target_mode.force_multi_obj()
        st_inter = State(
            mujoco_state=st.mujoco_state,
            tower_set=st.tower_set | frozenset([self.source_mode.multi_obj.next_obj_index]),
            tower_pos=st.tower_pos.copy(),
            goal_dict=dict(st.goal_dict),
        )
        self.target_mode.set_state(st)
        self.target_mode.multi_obj.goal = self.target_mode.multi_obj._sample_goal()
        return self.target_mode.get_state()
