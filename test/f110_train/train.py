import os
import sys
sys.path.append(os.path.join('..', '..'))  # nopep8

# flake8: noqa: E402
from hybrid_gym.train.single_mode import train_stable, make_sb_model
from hybrid_gym.train.mode_pred import train_mode_predictor
from hybrid_gym.util.wrappers import BaselineCtrlWrapper
from hybrid_gym.envs import make_f110_model


if __name__ == '__main__':
    automaton = make_f110_model(straight_lengths=[10])
    models = {
        name: make_sb_model(
            mode,
            automaton.transitions[name],
            algo_name='td3',
            action_noise_scale=8.0,
            verbose=2
        )
        for (name, mode) in automaton.modes.items()
    }
    for (name, mode) in automaton.modes.items():
        train_stable(models[name], mode, automaton.transitions[name],
                     total_timesteps=200000, algo_name='td3')
    controller = {name: BaselineCtrlWrapper(model) for (name, model) in models.items()}
    for (mode_name, ctrl) in controller.items():
        ctrl.save(f'{mode_name}.td3')
    # controller = {
    #     name: BaselineCtrlWrapper.load(f'{name}.td3', algo_name='td3')
    #     for name in automaton.modes
    # }
    # mode_pred = train_mode_predictor(
    #     automaton, {}, controller, 'mlp', num_iters=10,
    #     # hidden_layer_sizes=(192,32), activation='tanh'
    # )
    # mode_pred.save('mode_predictor.mlp')
