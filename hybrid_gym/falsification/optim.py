'''
Optimization methods used for falsification.
'''

import numpy as np

from typing import Callable, List
from hybrid_gym.synthesis.abstractions import AbstractState


def cem(f: Callable[[np.ndarray], float], X: AbstractState, mu: np.ndarray,
        sigma: np.ndarray, num_iter: int, samples_per_iter: int, num_top_samples: int,
        alpha: float = 0.9, print_debug: bool = False) -> List[np.ndarray]:
    '''
    Cross-entropy method for minimizing a function f.
    '''

    top_samples = []

    for i in range(num_iter):

        # initialize
        values: List[float] = []
        samples: List[np.ndarray] = []

        # collect samples
        while len(samples) < samples_per_iter:
            s = mu + sigma * np.random.randn(*mu.shape)
            if X.contains(s):
                samples.append(s)
        values = [f(s) for s in samples]

        # find top-k samples
        sorted_tuples = sorted(list(zip(values, list(range(samples_per_iter)))))
        top_tuples = sorted_tuples[:num_top_samples]
        top_samples = [samples[j] for _, j in top_tuples]

        # update mu and sigma
        mu_new = np.mean(top_samples, axis=0)
        sigma_new = np.std(top_samples, axis=0)
        mu = (1 - alpha) * mu_new + alpha * mu
        sigma = (1 - alpha) * sigma_new + alpha * sigma
        if print_debug:
            print('Iteration {}: mu = {} | sigma = {}'.format(i, mu, sigma))

    return top_samples
