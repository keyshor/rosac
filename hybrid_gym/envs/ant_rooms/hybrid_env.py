from hybrid_gym.hybrid_env import HybridAutomaton
from hybrid_gym.envs.ant_rooms.mode import AntMode, ModeType
from hybrid_gym.envs.ant_rooms.transition import AntTrans

def make_ant_model(**kwargs) -> HybridAutomaton:
    straight_mode = AntMode(mode_type=ModeType.STRAIGHT)
    left_mode = AntMode(mode_type=ModeType.LEFT)
    right_mode = AntMode(mode_type=ModeType.RIGHT)
    straight_trans = AntTrans(src_mt=ModeType.STRAIGHT)

    return HybridAutomaton(
        modes=[AntMode(mode_type=mt) for mt in ModeType],
        transitions=[AntTrans(src_mt=mt) for mt in ModeType],
    )
