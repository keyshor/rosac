import os
import sys
import gym
import numpy as np

sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8
from hybrid_gym.envs.ant_rooms.mode import AntMode, State, ModeType

def quat_mul(quat1, quat2):
    (t1, i1, j1, k1) = quat1
    (t2, i2, j2, k2) = quat2
    return (
        t1 * t2 - i1 * i2 - j1 * j2 - k1 * k2,
        t1 * i2 + i1 * t2 + j1 * k2 - k1 * j2,
        t1 * j2 + j1 * t2 - i1 * k2 + k1 * i2,
        t1 * k2 + k1 + t2 + i1 * j2 - j1 * i2,
    )

def quat_conj(quat):
    (t, i, j, k) = quat
    return (t, -i, -j, -k)

def rotatexy(angle, quat):
    angle_quat = (np.cos(angle), 0, 0, np.sin(angle))
    return quat_mul(angle_quat, quat)

def print_env_info():
    env = gym.make('Ant-v3')
    st = env.sim.get_state()
    #assert st.qpos.shape == (env.model.nq,) and st.qvel.shape == (env.model.nv,)
    #print(type(st.qpos))
    #print(type(st.qvel))
    #assert np.allclose(st.qpos, env.sim.data.qpos.flat)
    #assert np.allclose(st.qvel, env.sim.data.qvel.flat)
    #print(np.max(np.abs(st.qpos - env.sim.data.qpos.flat)))
    #print(np.max(np.abs(st.qvel - env.sim.data.qvel.flat)))
    env.reset()
    for _ in range(100):
        a = env.action_space.sample()
        obs, _, done, _ = env.step(a)
        xy_pos = env.get_body_com('torso')[:2]
        x_diff = obs[0] - xy_pos[0]
        y_diff = obs[1] - xy_pos[1]
        print(f'obs0 = {obs[0]}, obs1 = {obs[1]}, x = {xy_pos[0]}, y = {xy_pos[1]}, dx = {x_diff}, dy = {y_diff}')
        if done:
            break

if __name__ == '__main__':
    m = AntMode(mode_type=ModeType.STRAIGHT)
    print(m.observation_space.shape)
    #st = m.reset()
    #st.qpos[:2] = [0, -6]
    #for _ in range(10):
    #    a = m.action_space.sample()
    #    st = m.step(st, a)
    #print(st.qpos)
    #print(m.ant.get_body_com('torso'))
    #print(m.ant.sim.data.get_joint_qpos('root'))
    #print(st.qvel)
    #print(m.ant.sim.data.get_joint_qvel('root'))
    #zero_action = np.zeros(shape=m.action_space.shape)
    #st.qpos[3:7] = [0.96, 0.28, 0, 0]
    #for i in range(500):
    #    #st.qpos[3:7] = rotatexy(i / 500.0, (1, 0, 0, 0))
    #    st.qpos[0:3] = [0, 0, 2]
    #    st.qvel[3:6] = [0, 0, 1]
    #    #st.qvel[0:3] = [0, 0, 1]
    #    st = m.step(st, zero_action)
    #    m.render(st)
