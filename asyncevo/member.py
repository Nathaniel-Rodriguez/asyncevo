__all__ = ['Member']


import numpy as np
from asyncevo.basemember import BaseMember
from asyncevo.lineage import Lineage
from asyncevo.sliceops import *
from asyncevo import DEFAULT_TYPE


class Member(BaseMember):
    """
    Represents an expression of a lineage from the population. Upon
    initialization it generates the necessary allocations for appropriation
    of a lineage. After this, lineage appropriation is relatively cheap and
    the member can adopt new lineages, however it can only express a single
    lineage at a given time.
    """
    def __init__(self, initial_state: np.ndarray, table_seed: int,
                 table_size: int, max_table_step: int, dtype=DEFAULT_TYPE):
        """
        :param initial_state: a numpy array from which to generate perturbations.
        :param table_seed: the seed for generating the random number table.
        :param table_size: the size of the random number table.
        :param max_table_step: the maximum random stride for table slices
        """
        self._initial_state = initial_state
        self._x = np.copy(initial_state)
        self._mutation = np.zeros(len(self._x), dtype=dtype)
        self._rng = np.random.RandomState(table_seed)
        self._table = self._rng.randn(table_size)
        self._max_table_step = max_table_step
        self._has_lineage = False
        self._lineage = None

    @property
    def parameters(self):
        return self._x

    @parameters.setter
    def parameters(self, value):
        raise NotImplementedError

    def appropriate_lineage(self, lineage: Lineage):
        """
        Take on a lineage and explicitly express its parameters.
        :param lineage: a lineage to appropriate.
        """
        self._x[:] = self._initial_state[:]
        self._lineage = lineage
        self._has_lineage = True
        for mutation in self._lineage:
            self._rng.seed(mutation['seed'])
            param_slices = self._draw_random_parameter_slices(self._rng)
            table_slices = self._draw_random_table_slices(self._rng)
            param_slices, table_slices = match_slices(param_slices, table_slices)
            # We assign the table values to the perturbation member first
            multi_slice_assign(self._mutation, self._table,
                               param_slices, table_slices)
            # Apply the annealing scaling factor, i.e. the temperature
            np.multiply(self._mutation, mutation['sigma'],
                        out=self._mutation)
            # With perturbation member complete, we can add to member
            np.add(self._x, self._mutation, out=self._x)

    def _draw_random_parameter_slices(self, rng):
        """
        Chooses a constrained slice subset of the parameters (start, stop, step)
        to give roughly num_mutations perturbations (less if overlap if
        step is too large)
        """

        return random_slices(rng, len(self._x), len(self._x), 1)

    def _draw_random_table_slices(self, rng):
        """
        Chooses a constrained slice subset of the RN table (start, stop, step)
        to give roughly num_mutations random numbers (less if overlap if
        step is too large)
        """

        return random_slices(rng, len(self._table),
                             len(self._x), self._max_table_step)
