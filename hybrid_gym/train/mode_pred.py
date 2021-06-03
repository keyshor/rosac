import gym
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
from typing import (Tuple, Dict, Optional,
                    Callable, Any)
from typing_extensions import Protocol
from hybrid_gym.model import Controller, ModePredictor
from hybrid_gym.hybrid_env import HybridAutomaton


class ScipyClassifier(Protocol):
    def fit(self, data: np.ndarray, labels: np.ndarray) -> None:
        pass

    def predict(self, observations: np.ndarray) -> np.ndarray:
        pass


def scipy_from_str(model_type: str, **kwargs: Dict[str, Any]) -> ScipyClassifier:
    if model_type == 'svm':
        return make_pipeline(StandardScaler(), LinearSVC(**kwargs))
    if model_type == 'logistic':
        return make_pipeline(StandardScaler(), LogisticRegressionCV(**kwargs))
    if model_type == 'mlp':
        return make_pipeline(StandardScaler(), MLPClassifier(**kwargs))
    raise ValueError


class ScipyModePredictor(ModePredictor):
    model: ScipyClassifier
    observation_space: gym.Space

    def __init__(self, model: ScipyClassifier, observation_space: gym.Space) -> None:
        self.model = model
        self.observation_space = observation_space

    def fit(self, data: np.ndarray, labels: np.ndarray) -> None:
        self.model.fit(data, labels)

    def get_mode(self, observation: np.ndarray) -> str:
        assert self.observation_space.contains(observation)
        return self.model.predict(observation[np.newaxis, ...])[0]

    def save(self, path: str) -> None:
        # with open(path, 'w') as f:
        #    yaml.dump(self.model.get_params(), f)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str,
             observation_space: gym.Space,
             model_type: str = 'svm'
             ) -> ModePredictor:
        # m = scipy_from_str(model_type)
        # with open(path, 'r') as f:
        #    params = yaml.full_load(f)
        # m.set_params(**params)
        # return cls(m, observation_space)
        return joblib.load(path)


def generate_data(automaton: HybridAutomaton,
                  reset_fns: Dict[str, Callable[[], Any]],
                  controller: Dict[str, Controller],
                  mode_pred: Optional[ScipyClassifier] = None,
                  samples_per_mode: int = 1000,
                  max_episode_length: int = 200
                  ) -> Tuple[np.ndarray, np.ndarray]:
    if mode_pred is not None:
        def predict_mode(observation: np.ndarray, mode: str) -> str:
            assert mode_pred is not None  # for mypy
            return mode_pred.predict(observation[np.newaxis, ...])[0]
    else:
        def predict_mode(observation: np.ndarray, mode: str) -> str:
            return mode

    data = []
    labels = []
    for (mode_name, mode) in automaton.modes.items():
        reset_function = reset_fns.get(mode_name, mode.reset)
        st = reset_function()
        for i in range(samples_per_mode):
            observation = mode.observe(st)
            data.append(observation)
            labels.append(mode_name)
            pred = predict_mode(observation, mode_name)
            action = controller[pred].get_action(observation)
            st = mode.step(st, action)
            if not mode.is_safe(st) or \
                    any([t.guard(st) for t in automaton.transitions[mode_name]]):
                st = reset_function()
    data_array = np.array(data)
    label_array = np.array(labels)
    assert data_array.shape[0] == samples_per_mode * len(automaton.modes)
    assert label_array.shape == (samples_per_mode * len(automaton.modes),)
    return data_array, label_array


def train_mode_predictor(automaton: HybridAutomaton,
                         reset_fns: Dict[str, Callable[[], Any]],
                         controller: Dict[str, Controller],
                         model_type: str,
                         num_iters: int = 1,
                         samples_per_mode_per_iter: int = 1000,
                         max_episode_length: int = 200,
                         **scipy_kwargs: Dict[str, Any],
                         ) -> ModePredictor:
    mp: Optional[ScipyClassifier] = None
    data = np.zeros(tuple((0,) + automaton.observation_space.shape))
    labels = np.zeros(0)
    for _ in range(num_iters):
        new_data, new_labels = generate_data(
            automaton, reset_fns, controller, mp,
            samples_per_mode_per_iter, max_episode_length
        )
        data = np.concatenate([data, new_data])
        labels = np.concatenate([labels, new_labels])
        mp = scipy_from_str(model_type, **scipy_kwargs)
        mp.fit(data, labels)
    assert mp is not None
    return ScipyModePredictor(mp, automaton.observation_space)
