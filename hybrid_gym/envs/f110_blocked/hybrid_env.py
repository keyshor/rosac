import numpy as np
from hybrid_gym.hybrid_env import HybridAutomaton
from hybrid_gym.envs.f110.obstacle_mode import F110ObstacleMode
from hybrid_gym.envs.f110_blocked.mode import make_blocked
from hybrid_gym.envs.f110_blocked.transition import F110BlockedTrans

def make_f110_blocked_model(use_throttle: bool = True,
                            lidar_num_rays: int = 1081,
                            wide_width: float = 4.0,
                            narrow_width: float = 1.5,
                            length: float = 15,
                            observe_heading: bool = True,
                            supplement_start_region: bool = True,
                            observe_previous_lidar: bool = True,
                            ) -> HybridAutomaton:
    modes = [make_blocked(
        use_throttle=use_throttle, lidar_num_rays=lidar_num_rays,
        wide_width=wide_width, narrow_width=narrow_width, length=length,
        observe_heading=observe_heading, supplement_start_region=supplement_start_region,
        observe_previous_lidar=observe_previous_lidar,
    )]
    mode_dict = {mode.name: mode for mode in modes}
    return HybridAutomaton(
        modes=modes,
        transitions=[
            F110BlockedTrans(source=m.name, modes=modes)
            for m in modes
        ],
    )
