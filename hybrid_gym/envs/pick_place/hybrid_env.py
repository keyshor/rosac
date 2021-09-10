from hybrid_gym.hybrid_env import HybridAutomaton
from hybrid_gym.envs.pick_place.mode import PickPlaceMode, ModeType
from hybrid_gym.envs.pick_place.transition import PickPlaceTrans

from typing import List


def make_pick_place_model(num_objects: int = 3,
                          reward_type: str = 'sparse',
                          distance_threshold: float = 0.01,
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
        )
        move_with_obj = PickPlaceMode(
            mode_type=ModeType.MOVE_WITH_OBJ,
            next_obj_index = i,
            num_objects=num_objects,
            reward_type=reward_type,
            distance_threshold=distance_threshold,
        )
        pick_obj_pt1 = PickPlaceMode(
            mode_type=ModeType.PICK_OBJ_PT1,
            next_obj_index = i,
            num_objects=num_objects,
            reward_type=reward_type,
            distance_threshold=distance_threshold,
        )
        pick_obj_pt2 = PickPlaceMode(
            mode_type=ModeType.PICK_OBJ_PT2,
            next_obj_index = i,
            num_objects=num_objects,
            reward_type=reward_type,
            distance_threshold=distance_threshold,
        )
        pick_obj_pt3 = PickPlaceMode(
            mode_type=ModeType.PICK_OBJ_PT3,
            next_obj_index = i,
            num_objects=num_objects,
            reward_type=reward_type,
            distance_threshold=distance_threshold,
        )
        place_obj_pt1 = PickPlaceMode(
            mode_type=ModeType.PLACE_OBJ_PT1,
            next_obj_index = i,
            num_objects=num_objects,
            reward_type=reward_type,
            distance_threshold=distance_threshold,
        )
        place_obj_pt2 = PickPlaceMode(
            mode_type=ModeType.PLACE_OBJ_PT2,
            next_obj_index = i,
            num_objects=num_objects,
            reward_type=reward_type,
            distance_threshold=distance_threshold,
        )
        mv_wo_obj_modes.append(move_without_obj)
        place_pt2_modes.append(place_obj_pt2)
        modes += [
            move_without_obj,
            pick_obj_pt1, pick_obj_pt2, pick_obj_pt3,
            move_with_obj,
            place_obj_pt1, place_obj_pt2,
        ]
        transitions += [
            PickPlaceTrans(move_without_obj, pick_obj_pt1),
            PickPlaceTrans(pick_obj_pt1, pick_obj_pt2),
            PickPlaceTrans(pick_obj_pt2, pick_obj_pt3),
            PickPlaceTrans(pick_obj_pt3, move_with_obj),
            PickPlaceTrans(move_with_obj, place_obj_pt1),
            PickPlaceTrans(place_obj_pt1, place_obj_pt2),
        ]
    for place_obj_pt2 in place_pt2_modes:
        for move_without_obj in mv_wo_obj_modes:
            transitions.append(PickPlaceTrans(place_obj_pt2, move_without_obj))

    pick_place_automaton = HybridAutomaton(
        modes=modes,
        transitions=transitions,
    )

    return pick_place_automaton
