import os
import sys
sys.path.append(os.path.join('..', '..'))  # nopep8

from hybrid_gym.train.single_mode import train_stable, BaselineCtrlWrapper
from hybrid_gym.train.mode_pred import train_mode_predictor
from hybrid_gym.envs import make_pick_place_model

if __name__ == '__main__':
    automaton = make_pick_place_model(num_objects=3)
    controller = {
        name: train_stable(
            mode,
            automaton.transitions[name],
            algo_name='her',
            total_timesteps=200000,
            action_noise_scale=8.0,
            verbose=2
        )
        for (name, mode) in automaton.modes.items()
    }
    for (mode_name, ctrl) in controller.items():
        ctrl.save(f'{mode_name}.her')
    #controller = {
    #    name: BaselineCtrlWrapper.load(f'{name}.td3', algo_name='td3')
    #    for name in automaton.modes
    #}
    #mode_pred = train_mode_predictor(
    #    automaton, {}, controller, 'mlp', num_iters=10,
    #    #hidden_layer_sizes=(192,32), activation='tanh'
    #)
    #mode_pred.save('mode_predictor.mlp')
