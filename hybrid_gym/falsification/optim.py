'''
Optimization methods used for falsification.
'''

import numpy as np
from scipy.stats import truncnorm, norm

from typing import Callable, List, Tuple
from hybrid_gym.synthesis.abstractions import AbstractState, Box


def cem(f: Callable[[np.ndarray], Tuple[float, int]], X: AbstractState, mu: np.ndarray,
        sigma: np.ndarray, num_iter: int, samples_per_iter: int, num_top_samples: int,
        alpha: float = 0.9, print_debug: bool = False, max_samples=500
        ) -> Tuple[List[np.ndarray], int]:
    '''
    Cross-entropy method for minimizing a function f.
    '''

    top_samples = []
    steps_taken = 0

    for i in range(num_iter):

        # initialize
        values: List[float] = []
        samples: List[np.ndarray] = []

        # collect samples
        if isinstance(X, Box) and X.low is not None and X.high is not None:
            # Special case: abstract state is a box
            sgen = []
            for j in range(len(X.low)):
                sgen.append(_get_truncnorm(mu[j], sigma[j], X.low[j], X.high[j]))
            samples = [np.array([g.rvs() for g in sgen]) for _ in range(samples_per_iter)]
        else:
            # General case: use rejection sampling (slow)
            tries = 0
            while len(samples) < samples_per_iter:
                s = mu + sigma * np.random.randn(*mu.shape)
                if X.contains(s):
                    samples.append(s)
                tries += 1
                if tries >= max_samples:
                    break
        values = []
        for s in samples:
            v, steps = f(s)
            values.append(v)
            steps_taken += steps

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

    return top_samples, steps_taken


def _get_truncnorm(mean, sd, low, high):
    a = (low-mean)/sd
    b = (high-mean)/sd
    assert sd > 0, f'invalid standard deviation {sd}'
    if abs(a-b) < 1e-15:
        return norm(loc=mean, scale=0.)
    return truncnorm(a, b, loc=mean, scale=sd)
