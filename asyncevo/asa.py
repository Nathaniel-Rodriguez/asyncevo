__all__ = ['AdaptiveSimulatedAnnealing']


import math
import numpy as np


class AdaptiveSimulatedAnnealing:
    """
    Implements a cooling schedule based on the adaptive schedule developed by:
    Huang, M.D., Romeo, F., Sangiovanni-Vincentelli, A.L., 1986.
    An efficient general cooling schedule for simulated annealing,
    In: Proceedings of the IEEE International Conference on
    Computer-Aided Design, Santa Clara, pp. 381–384.

    This cooling schedule will slow down the cooling rate as the variance of
    the energy increases, and speed it up if the variance decreases.

    This cooling schedule is designed to work in environment where samples
    are accumulated asynchronously. When the algorithm is stepped it takes a
    temperature and energy of a sample and then makes an incremental weighted
    adjustment to the current temperature. Samples from temperatures very
    different from the current temperature, but close in time, are weighted
    low, while temperatures that are close to the current temperature are
    weighted highly. These energies are used to estimate the heat capacity
    of the system, which is then used by the referenced cooling schedule to
    update the temperature.
    """
    def __init__(self,
                 t0: float,
                 cooling_factor: float,
                 tmin: float = 0.0,
                 max_estimate_window: int = 10000,
                 decay_factor: float = 1.0,
                 hold_window: int = 100):
        """
        :param t0: initial temperature
        :param cooling_factor: determines how quickly the cooling rate can
        change. A low value means fast change, while a high value means
        changes will be slow. Value is bound between [0,1].
        :param tmin: the minimum allowed temperature. Default 0.0
        :param max_estimate_window: the maximum allowed history to record.
        :param decay_factor: how strongly to penalize energy contributions from
        temperatures different from the current temperature when estimating
        the heat capacity.
        :param hold_window: how many initial steps to wait before updating the
        temperature. Since one sample is gained each step, this equates to the
        number of samples that will be used to generate the first heat
        capacity estimate.
        """

        self._t0 = t0  # initial temperature
        self._tc = self._t0  # current temperature
        self._g = cooling_factor
        self._tmin = tmin  # minimum temperature
        self._decay_factor = decay_factor
        self._t_log = [self._t0]  # log of annealed temperatures
        self._hold_window = hold_window  # how long to wait till t updates
        self._step_count = 0

        self._t_history = np.zeros(max_estimate_window)  # sample temp history
        self._e_history = np.zeros(max_estimate_window)  # sample energy history
        self._weights = np.zeros(max_estimate_window)  # sample weights
        self._var_buffer = np.zeros(max_estimate_window)  # pre-allocated buffer
        self._sample_index = max_estimate_window  # index for first valid sample

    def step(self, sample_t: float, sample_e: float) -> float:
        """
        Steps the cooling schedule.
        :param sample_t: temperature of the given sample
        :param sample_e: energy of the given sample
        :return: the new temperature
        """
        if self._sample_index > 0:
            self._sample_index -= 1

        # pop the old sample and add the newest
        self._t_history[:-1] = self._t_history[1:]
        self._t_history[-1] = sample_t
        self._e_history[:-1] = self._e_history[1:]
        self._e_history[-1] = sample_e

        # wait to update the temperature until enough samples are accumulated
        if self._step_count < self._hold_window:
            self._step_count += 1
            return self._tc
        elif self._tc >= self._tmin:
            return self._update_temperature()
        else:
            return self._tc

    def _update_temperature(self) -> float:
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
