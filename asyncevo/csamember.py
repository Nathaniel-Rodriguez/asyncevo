__all__ = ['CSAMember', 'DiagnosticCSAMember']


import numpy as np
import math
from asyncevo.csalineage import CSALineage
from asyncevo.sliceops import *
from asyncevo.basemember import BaseMember
from asyncevo import DEFAULT_TYPE


class CSAMember(BaseMember):
    """
    Represents an expression of a lineage from the population for an
    AsyncCSAGa. Upon initialization it generates the necessary allocations
    for appropriation of a lineage. After this, lineage appropriation is
    relatively cheap and the member can adopt new lineages, however it can
    only express a single lineage at a given time.
    """
    def __init__(self, initial_state: np.ndarray,
                 initial_sigma: np.ndarray, table_seed: int,
                 table_size: int, max_table_step: int,
                 path_memory: float, adaptation_speed: float,
                 adaptation_precision: float, population_size: int,
                 dtype=DEFAULT_TYPE):
        """
        :param initial_state: a numpy array from which to generate perturbations.
        :param table_seed: the seed for generating the random number table.
        :param table_size: the size of the random number table.
        :param max_table_step: the maximum random stride for table slices
        """
        self._initial_state = initial_state
        self._initial_sigma = initial_sigma
        self._x = np.copy(initial_state)
        self._sigma = np.copy(initial_sigma)
        self._path = np.zeros(len(self._x), dtype=dtype)
        self._delta_path = np.zeros(len(self._x), dtype=dtype)
        self._path_buffer = self._delta_path
        self._mutation = np.zeros(len(self._x), dtype=dtype)
        self._rng = np.random.RandomState(table_seed)
        self._table = self._rng.randn(table_size)
        self._max_table_step = max_table_step
        self._path_memory = path_memory
        self._adaptation_speed = adaptation_speed
        self._adaptation_precision = adaptation_precision
        self._population_size = population_size
        self._weights = np.log(self._population_size + 0.5) - np.log(
            np.arange(1, self._population_size + 1))
        self._weights /= np.sum(self._weights)
        self._weights.astype(np.float32, copy=False)
        self._weight_scale = 1. / np.sum(self._weights**2)
        self._path_rescaling_factor = math.sqrt(self._path_memory
                                                * (2 - self._path_memory))\
                                      * math.sqrt(self._weight_scale)
        # this is E(|N(0,1)|) where | is the absolute value and E is expected
        self._abs_norm_factor = math.sqrt(2 / math.pi)
        self._memory_factor = 1 - self._path_memory
        self._has_lineage = False
        self._lineage = None
        self._global_sigma = 0

    @property
    def parameters(self):
        return self._x

    @parameters.setter
    def parameters(self, value):
        raise NotImplementedError

    @property
    def data(self):
        return None

    def appropriate_lineage(self, lineage: CSALineage):
        """
        Take on a lineage and explicitly express its parameters.
        :param lineage: a lineage to appropriate.
        """
        # reset local variables to initial state
        self._x[:] = self._initial_state[:]
        self._sigma[:] = self._initial_sigma[:]
        self._path.fill(0.0)
        self._lineage = lineage
        self._has_lineage = True

        for i in range(len(self._lineage.lineage)):
            self._regenerate_mutation(i)
            self._regenerate_parameters()

            if i < (len(self._lineage.lineage) - 1):
                self._regenerate_path(i)
                self._regenerate_sigma()

    def _regenerate_path(self, historical_index: int):
        """
        s_path = (1 - path_memory) * s_path + path_rescaling_factor * sum(mutation * weight)
        """
        pop = self._lineage.path[historical_index]
        self._delta_path.fill(0.0)
        # add up all the permutations to find delta path
        for i, member in enumerate(sorted(pop, key=lambda x: x.fitness,
                                          reverse=True)):
            self._rng.seed(member.seed)
            param_slices = self._draw_random_parameter_slices(self._rng)
            table_slices = self._draw_random_table_slices(self._rng)
            param_slices, table_slices = match_slices(param_slices, table_slices)
            # We assign the table values to the perturbation member first
            multi_slice_assign(self._mutation, self._table,
                               param_slices, table_slices)
            np.multiply(self._mutation, self._weights[i], out=self._mutation)
            np.add(self._delta_path, self._mutation, out=self._delta_path)

        # update the current path
        np.multiply(self._delta_path, self._path_rescaling_factor,
                    out=self._delta_path)
        np.multiply(self._path, self._memory_factor, out=self._path)
        np.add(self._path, self._delta_path, out=self._path)

    def _regenerate_sigma(self):
        """
        Here sqrt(2/PI) stands in for univariate E(|N(0,1)|) and sqrt(N) stands
        in for multivariate E(||N(0,1)||)
        N == dimensionality of x, # of parameters
        a = exp(path_memory / adaptation_speed * ( ||path|| / sqrt(N) - 1 )
        b = exp(1 / adaptation_precision * ( |path| / sqrt(2 / PI) - 1 )
        sigma = sigma * a * b
        a will be a scalar representing the global step size and
        b will be a vector representing parameter-wise step sizes
        """
        # global step size update
        self._path_buffer[:] = self._path[:]
        np.square(self._path_buffer, out=self._path_buffer)
        global_sigma = math.exp(self._path_memory / self._adaptation_speed
                                * (math.sqrt(np.sum(self._path_buffer))
                                   / math.sqrt(len(self._x)) - 1))
        self._global_sigma = global_sigma

        # parameter-wise step size update
        self._path_buffer[:] = self._path[:]
        np.fabs(self._path_buffer, out=self._path_buffer)
        np.divide(self._path_buffer, self._abs_norm_factor, out=self._path_buffer)
        np.subtract(self._path_buffer, 1, out=self._path_buffer)
        np.multiply(1. / self._adaptation_precision,
                    self._path_buffer, out=self._path_buffer)
        np.exp(self._path_buffer, out=self._path_buffer)

        # update sigma
        np.multiply(self._sigma, self._path_buffer, out=self._sigma)
        np.multiply(self._sigma, global_sigma, out=self._sigma)

    def _regenerate_mutation(self, historical_index: int):
        self._rng.seed(self._lineage.lineage[historical_index])
        param_slices = self._draw_random_parameter_slices(self._rng)
        table_slices = self._draw_random_table_slices(self._rng)
        param_slices, table_slices = match_slices(param_slices, table_slices)
        # We assign the table values to the perturbation member first
        multi_slice_assign(self._mutation, self._table,
                           param_slices, table_slices)

    def _regenerate_parameters(self):
        np.multiply(self._mutation, self._sigma,
                    out=self._mutation)
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


class DiagnosticCSAMember(CSAMember):
    """
    This diagnostic class adds a data property which returns four indicators
    of the CSA member:
        global_sigma - the value of the global component of the step-size update.
        abs_sigma - the mean of the absolute values of the sigma vector.
        abs_path - the mean of the absolute values of the path vector. This is
            most closely related to the local component of the step-size updated.
        path_norm - the norm of the path vector. This is most closely related
            to the global component of the step-size update.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def data(self):
        np.fabs(self._path, out=self._path_buffer)
        abs_path = np.mean(self._path_buffer)
        path_norm = np.linalg.norm(self._path)
        np.fabs(self._sigma, out=self._path_buffer)
        abs_sigma = np.mean(self._path_buffer)
        return {'global_sigma': self._global_sigma,
                'abs_sigma': abs_sigma,
                'abs_path': abs_path,
                'path_norm': path_norm}
