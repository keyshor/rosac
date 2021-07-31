import os
import sys
sys.path.append(os.path.join('..', '..'))  # nopep8
sys.path.append(os.path.join('..', '..', 'spectrl_hierarchy'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.train.single_mode import train_stable, make_sb_model
from hybrid_gym.train.mode_pred import train_mode_predictor
from hybrid_gym.util.wrappers import BaselineCtrlWrapper
from hybrid_gym.envs import make_f110_model


automaton = make_f110_model(straight_lengths=[10])


def train_single(name, total_timesteps, save_path):
    mode = automaton.modes[name]
    model = make_sb_model(
        mode,
        automaton.transitions[name],
        algo_name='td3',
        action_noise_scale=4.0,
        verbose=0,
        max_episode_steps=100,
    )
    train_stable(model, mode, automaton.transitions[name],
                 total_timesteps=total_timesteps, algo_name='td3',
                 max_episode_steps=100, save_path=save_path)


def train_all_modes(save_path):
    automaton = make_f110_model(straight_lengths=[10])
    models = {
        name: make_sb_model(
            mode,
            automaton.transitions[name],
            algo_name='td3',
            action_noise_scale=4.0,
            verbose=0,
            max_episode_steps=100,
        )
        for (name, mode) in automaton.modes.items()
    }
    for (name, mode) in automaton.modes.items():
        train_stable(models[name], mode, automaton.transitions[name],
                     total_timesteps=20000, algo_name='td3', save_path=save_path)


def train_mode_pred(save_path):
    controller = {
        name: BaselineCtrlWrapper.load(os.path.join(
            save_path, name, 'best_model.zip'), algo_name='td3')
        for name in automaton.modes
    }
    mode_pred = train_mode_predictor(
        automaton, {}, controller, 'mlp', num_iters=10,
        # hidden_layer_sizes=(192,32), activation='tanh'
    )
    mode_pred.save(os.path.join(save_path, 'mode_predictor.mlp'))


if __name__ == '__main__':
    save_path = '.'
    if len(sys.argv) >= 2:
        save_path = sys.argv[1]
        mode_list = sys.argv[2:]
    else:
        mode_list = list(automaton.modes)
    for name in mode_list:
        print(f'training mode {name}')
        train_single(name, 200000, save_path)
