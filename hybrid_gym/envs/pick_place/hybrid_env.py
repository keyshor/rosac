from hybrid_gym.hybrid_env import HybridAutomaton
from hybrid_gym.envs.pick_place.mode import PickPlaceMode, ModeType
from hybrid_gym.envs.pick_place.transition import PickPlaceTrans

from typing import List


def make_pick_place_model(num_objects: int = 3,
                          reward_type: str = 'sparse',
                          distance_threshold: float = 0.01,
                          fixed_tower_height: bool = False,
                          flatten_obs: bool = False,
                          ) -> HybridAutomaton:
    modes = []
    transitions = []
    place_pt2_modes: List[PickPlaceMode] = []
    mv_wo_obj_modes: List[PickPlaceMode] = []
    for i in range(num_objects):
        move_without_obj = PickPlaceMode(
            mode_type=ModeType.MOVE_WITHOUT_OBJ,
            next_obj_index = i,
            num_objects=num_objects,
            reward_type=reward_type,
            distance_threshold=distance_threshold,
            flatten_obs=flatten_obs,
        )
        move_with_obj_modes = [
            PickPlaceMode(
                mode_type=ModeType.MOVE_WITH_OBJ,
                next_obj_index = i,
                num_objects=num_objects,
                fixed_tower_height=j,
                reward_type=reward_type,
                distance_threshold=distance_threshold,
                flatten_obs=flatten_obs,
            )
            for j in range(num_objects)
        ] if fixed_tower_height else [
            PickPlaceMode(
                mode_type=ModeType.MOVE_WITH_OBJ,
                next_obj_index = i,
                num_objects=num_objects,
                fixed_tower_height=None,
                reward_type=reward_type,
                distance_threshold=distance_threshold,
                flatten_obs=flatten_obs,
            )
        ]
        pick_obj_pt1 = PickPlaceMode(
            mode_type=ModeType.PICK_OBJ_PT1,
            next_obj_index = i,
            num_objects=num_objects,
            reward_type=reward_type,
            distance_threshold=distance_threshold,
            flatten_obs=flatten_obs,
        )
        pick_obj_pt2 = PickPlaceMode(
            mode_type=ModeType.PICK_OBJ_PT2,
            next_obj_index = i,
            num_objects=num_objects,
            reward_type=reward_type,
            distance_threshold=distance_threshold,
            flatten_obs=flatten_obs,
        )
        pick_obj_pt3 = PickPlaceMode(
            mode_type=ModeType.PICK_OBJ_PT3,
            next_obj_index = i,
            num_objects=num_objects,
            reward_type=reward_type,
            distance_threshold=distance_threshold,
            flatten_obs=flatten_obs,
        )
        place_obj_pt1 = PickPlaceMode(
            mode_type=ModeType.PLACE_OBJ_PT1,
            next_obj_index = i,
            num_objects=num_objects,
            reward_type=reward_type,
            distance_threshold=distance_threshold,
            flatten_obs=flatten_obs,
        )
        place_obj_pt2 = PickPlaceMode(
            mode_type=ModeType.PLACE_OBJ_PT2,
            next_obj_index = i,
            num_objects=num_objects,
            reward_type=reward_type,
            distance_threshold=distance_threshold,
            flatten_obs=flatten_obs,
        )
        mv_wo_obj_modes.append(move_without_obj)
        place_pt2_modes.append(place_obj_pt2)
        modes += [
            move_without_obj,
            pick_obj_pt1, pick_obj_pt2, pick_obj_pt3,
            place_obj_pt1, place_obj_pt2,
        ]
        modes += move_with_obj_modes
        transitions += [
            PickPlaceTrans(move_without_obj, pick_obj_pt1),
            PickPlaceTrans(pick_obj_pt1, pick_obj_pt2),
            PickPlaceTrans(pick_obj_pt2, pick_obj_pt3),
            PickPlaceTrans(place_obj_pt1, place_obj_pt2),
        ]
        transitions += [
            PickPlaceTrans(pick_obj_pt3, move_with_obj_m)
            for move_with_obj_m in move_with_obj_modes
        ]
        transitions += [
            PickPlaceTrans(move_with_obj_m, place_obj_pt1)
            for move_with_obj_m in move_with_obj_modes
        ]
    for place_obj_pt2 in place_pt2_modes:
        for move_without_obj in mv_wo_obj_modes:
            transitions.append(PickPlaceTrans(place_obj_pt2, move_without_obj))

    pick_place_automaton = HybridAutomaton(
        modes=modes,
        transitions=transitions,
    )

    return pick_place_automaton
