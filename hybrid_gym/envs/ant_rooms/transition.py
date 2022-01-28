import numpy as np
from hybrid_gym.envs.ant_rooms.mode import ModeType, State
from hybrid_gym.model import Transition
from typing import Tuple

Quat = Tuple[float, float, float, float]

def quat_mul(quat1: Quat, quat2: Quat) -> Quat:
    (t1, i1, j1, k1) = quat1
    (t2, i2, j2, k2) = quat2
    return (
        t1 * t2 - i1 * i2 - j1 * j2 - k1 * k2,
        t1 * i2 + i1 * t2 + j1 * k2 - k1 * j2,
        t1 * j2 + j1 * t2 - i1 * k2 + k1 * i2,
        t1 * k2 + k1 + t2 + i1 * j2 - j1 * i2,
    )

SQRT_HALF: float = float(np.sqrt(0.5))
QUAT_POS90: Quat = (SQRT_HALF, 0.0, 0.0, SQRT_HALF)
QUAT_NEG90: Quat = (SQRT_HALF, 0.0, 0.0, -SQRT_HALF)

class AntTrans(Transition):
    src_mt: ModeType

    def __init__(self,
                 src_mt: ModeType,
                 ) -> None:
        self.src_mt = src_mt
        super().__init__(src_mt.name, [mt.name for mt in ModeType])

    def guard(self, st: State) -> bool:
        return (
            (self.src_mt == ModeType.STRAIGHT and st.info['y_position'] >= 6.0)
            or (self.src_mt == ModeType.LEFT and st.info['x_position'] <= -6.0)
            or (self.src_mt == ModeType.RIGHT and st.info['x_position'] >= 6.0)
        )

    def jump(self, target: str, st: State) -> State:
        qpos = np.array(st.qpos)
        qvel = np.array(st.qvel)
        if self.src_mt == ModeType.STRAIGHT:
            qpos[1] -= 12.0
        elif self.src_mt == ModeType.LEFT:
            qpos[:2] = [qpos[1], -qpos[0] - 12.0]
            qvel[:2] = [qvel[1], -qvel[0] - 12.0]
            qpos[3:7] = quat_mul(
                QUAT_NEG90, (qpos[3], qpos[4], qpos[5], qpos[6]),
            )
        elif self.src_mt == ModeType.RIGHT:
            qpos[:2] = [-qpos[1], qpos[0] - 12.0]
            qvel[:2] = [-qvel[1], qvel[0] - 12.0]
            qpos[3:7] = quat_mul(
                QUAT_POS90, (qpos[3], qpos[4], qpos[5], qpos[6]),
            )
        return State(
            qpos=qpos,
            qvel=qvel,
            observation=st.observation,
            info=st.info,
            is_healthy=st.is_healthy,
        )
