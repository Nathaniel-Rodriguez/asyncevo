__all__ = ['AsyncCSAGa']

import numpy as np
from typing import Dict
from pathlib import Path
from random import Random
from math import inf
from math import sqrt
from copy import deepcopy
from asyncevo import Scheduler
from asyncevo import CSALineage
from asyncevo import CSAMember
from asyncevo import split_work
from asyncevo import save
from asyncevo import load
from asyncevo import DEFAULT_TYPE
from asyncevo import initialize_member
from asyncevo import dispatch_work


class AsyncCSAGa:
    """
    An asynchronous stead-state genetic algorithm that uses random selection
    and worst replacement. Additionally, it uses cumulative step-size
    adaptation (CSA) which uses parameter-wise evolutionary paths to adapt
    the step-size of the mutations over time.
    """
    dtype = DEFAULT_TYPE

    def __init__(self,
                 initial_state: np.ndarray,
                 initial_sigma: np.ndarray,
                 population_size: int,
                 scheduler: Scheduler,
                 global_seed: int,
                 path_memory: float=None,
                 adaptation_speed: float=None,
                 adaptation_precision: float=None,
                 table_size: int = 20000000,
                 max_table_step: int = 5,
                 member_type=CSAMember,
                 member_type_kwargs: Dict = None,
                 save_filename: Path = None,
                 save_every: int = None,
                 *args,
                 **kwargs):

        if member_type_kwargs is None:
            member_type_kwargs = {}

        if path_memory is None:
            path_memory = sqrt(population_size /
                               (len(initial_state) + population_size))

        if adaptation_speed is None:
            adaptation_speed = 1 + sqrt(population_size / len(initial_state))

        if adaptation_precision is None:
            adaptation_precision = 3 * len(initial_state)

        self._initial_state = initial_state.astype(dtype=AsyncCSAGa.dtype)
        self._initial_sigma = initial_sigma.astype(dtype=AsyncCSAGa.dtype)
        self._population_size = population_size
        self._scheduler = scheduler
        self._global_seed = global_seed
        self._step = 0
        self._path_memory = path_memory
        self._adaptation_speed = adaptation_speed
        self._adaptation_precision = adaptation_precision
        self._table_size = table_size
        self._max_table_step = max_table_step
        self._save_filename = save_filename
        self._save_every = save_every
        self._from_file = kwargs.get("from_file", False)

        self._cost_rank_sum = self._population_size \
                              * (self._population_size + 1) / 2
        self._selection_probabilities = [self._linearly_scaled_member_rank(i)
                                         for i in range(self._population_size)]
        self._fitness_history = kwargs.get('history', [])
        self._rng = Random(self._global_seed)
        self._population = kwargs.get('population', self._initialize())
        self._table_seed = kwargs.get('table_seed', self._make_seed())
        self._member_type = member_type
        self._member_type_kwargs = member_type_kwargs

    @classmethod
    def from_file(cls,
                  filename: Path,
                  scheduler: Scheduler,
                  global_seed: int,
                  member_type=CSAMember,
                  member_type_kwargs: Dict = None,
                  save_filename: Path = None,
                  save_every: int = None):
        file_contents = load(filename)
        return cls(file_contents['initial_state'],
                   file_contents['initial_sigma'],
                   len(file_contents['population']),
                   scheduler,
                   global_seed,
                   file_contents['path_memory'],
                   file_contents['adaptation_speed'],
                   file_contents['adaptation_precision'],
                   file_contents['table_size'],
                   file_contents['max_table_step'],
                   member_type,
                   member_type_kwargs,
                   save_filename,
                   save_every,
                   population=file_contents['population'],
                   history=file_contents['history'],
                   table_seed=file_contents['table_seed'],
                   from_file=True)

    def _linearly_scaled_member_rank(self, cost_index):
        """
        Scales the rank of an individual (cost_index)
        :param cost_index: 1 is best
        :return: scaled_cost_rank
        """
        return (self._population_size - cost_index) / self._cost_rank_sum

    def save_population(self, filename: Path):
        """
        Saves the current population to file along with all necessary parameters
        required for re-creating the base Member class that generates the
        parameter vector. Also included is the whole history of fitness values
        in the population. The data is saved as a pickled Python object.
        Compatible versions of Numpy should be used as Numpy objects are also
        pickled. The data layout is follows:

            {
             'population': List[{'lineage': Lineage, 'fitness': float}],
             'initial_state': np.ndarray,
             'initial_sigma': np.ndarray,
             'path_memory': float,
             'adaptation_speed': float,
             'adaptation_precision': float,
             'table_seed': int,
             'table_size': int,
             'max_table_step': int,
             'history': List[List['fitness]]
             }

        :param filename: a file path
        """
        save({
            'population': self._population,
            'initial_state': self._initial_state,
            'initial_sigma': self._initial_sigma,
            'path_memory': self._path_memory,
            'adaptation_speed': self._adaptation_speed,
            'adaptation_precision': self._adaptation_precision,
            'table_seed': self._table_seed,
            'table_size': self._table_size,
            'max_table_step': self._max_table_step,
            'history': self._fitness_history
        }, filename)
