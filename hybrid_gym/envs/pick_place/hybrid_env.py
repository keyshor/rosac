from hybrid_gym.hybrid_env import HybridAutomaton
from hybrid_gym.envs.pick_place.mode import PickPlaceMode, ModeType
from hybrid_gym.envs.pick_place.transition import PickPlaceTrans


def make_pick_place_model(num_objects: int = 3,
                          reward_type: str = 'sparse',
                          distance_threshold: float = 0.01,
                          ) -> HybridAutomaton:
    move_without_obj = PickPlaceMode(
        mode_type=ModeType.MOVE_WITHOUT_OBJ,
        num_objects=num_objects,
        reward_type=reward_type,
        distance_threshold=distance_threshold,
    )
    move_with_obj = PickPlaceMode(
        mode_type=ModeType.MOVE_WITH_OBJ,
        num_objects=num_objects,
        reward_type=reward_type,
        distance_threshold=distance_threshold,
    )
    pick_obj_pt1 = PickPlaceMode(
        mode_type=ModeType.PICK_OBJ_PT1,
        num_objects=num_objects,
        reward_type=reward_type,
        distance_threshold=distance_threshold,
    )
    pick_obj_pt2 = PickPlaceMode(
        mode_type=ModeType.PICK_OBJ_PT2,
        num_objects=num_objects,
        reward_type=reward_type,
        distance_threshold=distance_threshold,
    )
    pick_obj_pt3 = PickPlaceMode(
        mode_type=ModeType.PICK_OBJ_PT3,
        num_objects=num_objects,
        reward_type=reward_type,
        distance_threshold=distance_threshold,
    )
    place_obj_pt1 = PickPlaceMode(
        mode_type=ModeType.PLACE_OBJ_PT1,
        num_objects=num_objects,
        reward_type=reward_type,
        distance_threshold=distance_threshold,
    )
    place_obj_pt2 = PickPlaceMode(
        mode_type=ModeType.PLACE_OBJ_PT2,
        num_objects=num_objects,
        reward_type=reward_type,
        distance_threshold=distance_threshold,
    )

    pick_place_automaton = HybridAutomaton(
        modes=[move_without_obj, pick_obj_pt1, pick_obj_pt2, pick_obj_pt3, move_with_obj, place_obj_pt1, place_obj_pt2],
        transitions=[
            PickPlaceTrans(move_without_obj, pick_obj_pt1),
            PickPlaceTrans(pick_obj_pt1, pick_obj_pt2),
            PickPlaceTrans(pick_obj_pt2, pick_obj_pt3),
            PickPlaceTrans(pick_obj_pt3, move_with_obj),
            PickPlaceTrans(move_with_obj, place_obj_pt1),
            PickPlaceTrans(place_obj_pt1, place_obj_pt2),
            PickPlaceTrans(place_obj_pt2, move_without_obj)])

    return pick_place_automaton
