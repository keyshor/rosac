from hybrid_gym.hybrid_env import HybridAutomaton, HybridEnv
from hybrid_gym.selectors import UniformSelector
from hybrid_gym.envs.f110.mode import F110Mode, normal_right, normal_left, sharp_right, sharp_left
from hybrid_gym.envs.f110.transition import straight_10, straight_15, F110Trans
from typing import List

modes: List[F110Mode] = [
    straight_10,
    straight_15,
    normal_right,
    normal_left,
    sharp_right,
    sharp_left,
]

f110_automaton: HybridAutomaton = HybridAutomaton(
    modes=modes,
    transitions=[
        F110Trans(source=src.name, targets=[tgt.name for tgt in modes])
        for src in modes
    ]
)

f110_env = HybridEnv(
    automaton=f110_automaton,
    selector=UniformSelector(modes),
)
