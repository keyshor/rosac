from hybrid_gym.hybrid_env import HybridAutomaton
from hybrid_gym.envs.ant_rooms.mode import AntMode, ModeType
from hybrid_gym.envs.ant_rooms.transition import AntTrans

def make_ant_model(**kwargs) -> HybridAutomaton:
    return HybridAutomaton(
        modes=[AntMode(mode_type=mt, **kwargs) for mt in ModeType],
        transitions=[AntTrans(src_mt=mt) for mt in ModeType],
    )
