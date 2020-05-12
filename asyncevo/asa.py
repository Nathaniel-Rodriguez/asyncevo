__all__ = ['AdaptiveSimulatedAnnealing']


import math
import numpy as np


class AdaptiveSimulatedAnnealing:
    """
    Huang, M.D., Romeo, F., Sangiovanni-Vincentelli, A.L., 1986.
    An efficient general cooling schedule for simulated annealing,
    In: Proceedings of the IEEE International Conference on
    Computer-Aided Design, Santa Clara, pp. 381–384.
    """
    def __init__(self,
                 t0: float,
                 cooling_factor: float,
                 tmin: float = 0.0,
                 max_estimate_window: int = 10000,
                 decay_factor: float = 1.0):
        self._t0 = t0  # initial temperature
        self._tc = self._t0  # current temperature
        self._g = cooling_factor
        self._tmin = tmin  # minimum temperature
        self._decay_factor = decay_factor
        self._t_log = [self._t0]  # log of annealed temperatures

        self._t_history = np.zeros(max_estimate_window)  # sample temp history
        self._e_history = np.zeros(max_estimate_window)  # sample energy history
        self._weights = np.zeros(max_estimate_window)  # sample weights
        self._var_buffer = np.zeros(max_estimate_window)  # pre-allocated buffer
        self._sample_index = max_estimate_window  # index for first valid sample

    def step(self, sample_t: float, sample_e: float) -> float:
        if self._sample_index > 0:
            self._sample_index -= 1

        # pop the old sample and add the newest
        self._t_history[:-1] = self._t_history[1:]
        self._t_history[-1] = sample_t
        self._e_history[:-1] = self._e_history[1:]
        self._e_history[-1] = sample_e

        # update weights
        np.exp(
            np.multiply(
                self._decay_factor,
                np.subtract(self._tc, self._t_history, out=self._weights),
                out=self._weights),
            out=self._weights)

        # calculate energy standard deviation
        weighted_e_mean = np.average(self._e_history[self._sample_index:],
                                     weights=self._weights[self._sample_index:])
        weighted_e_std = math.sqrt(np.average(
            np.square(
                np.subtract(self._e_history[self._sample_index:],
                            weighted_e_mean,
                            out=self._var_buffer[self._sample_index:]),
                out=self._var_buffer[self._sample_index:]),
            weights=self._weights[self._sample_index:]
        ) / self._weights[self._sample_index:].sum())

        # update temperature
        t_new = self._tc * math.exp(-self._g * self._tc / weighted_e_std)
        self._t_log.append(t_new)
        return t_new
