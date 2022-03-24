import os
import sys
import argparse
import pathlib
import pickle
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.util.wrappers import GymEnvWrapper
from hybrid_gym.envs.ant_rooms.hybrid_env import make_ant_model
from hybrid_gym.rl.sac.sac import SACController
from hybrid_gym.hybrid_env import HybridEnv
from hybrid_gym.selectors import UniformSelector, MaxJumpWrapper

import matplotlib.pyplot as plt


def generate_rollouts(automaton, src_mode, tgt_mode, time_limit, num_rollouts, model_path, data_path):
    mode = automaton.modes[src_mode]
    trans = automaton.transitions[src_mode][0]
    env = GymEnvWrapper(automaton, mode, automaton.transitions[src_mode])
    controller = SACController.load(
        os.path.join(model_path, f'{src_mode}_final.pkl'),
    )
    sts = []
    while len(sts) < num_rollouts:
        observation = env.reset()
        e = 0
        done = False
        while not done:
            if e > time_limit:
                break
            e += 1
            action = controller.get_action(observation)
            observation, _, done, _ = env.step(action)
            # print(reward)
        if e <= time_limit and env.mode.is_safe(env.state):
            sts.append(trans.jump(tgt_mode, env.state))
    with open(data_path, 'wb') as fh:
        pickle.dump(sts, fh)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='generate data for tests')
    ap.add_argument('--model-path', type=pathlib.Path, default='.',
                    help='directory in which models are stored')
    ap.add_argument('--data-path', type=pathlib.Path, default='data.pkl',
                    help='place to store generated data')
    ap.add_argument('--num-rollouts', type=int, default=10,
                    help='number of rollouts')
    ap.add_argument('--mode-length', type=int, default=1000,
                    help='maximum number of time-steps in each mode')
    ap.add_argument('--src-mode', type=str, default='STRAIGHT',
                    help='source mode for data generation')
    ap.add_argument('--tgt-mode', type=str, default='STRAIGHT',
                    help='target mode for data generation')
    args = ap.parse_args()

    automaton = make_ant_model()
    generate_rollouts(automaton, args.src_mode, args.tgt_mode, args.mode_length, args.num_rollouts, args.model_path, args.data_path)
    with open(args.data_path, 'rb') as fh:
        sts = pickle.load(fh)
    assert len(sts) == args.num_rollouts
