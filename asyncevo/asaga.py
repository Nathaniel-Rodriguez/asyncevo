__all__ = ['ASAGa']


import numpy as np
from math import inf
from typing import Dict
from typing import Callable
from typing import List
from copy import deepcopy
from pathlib import Path
from random import Random
from asyncevo import Member
from asyncevo import Scheduler
from asyncevo import Lineage
from asyncevo import DEFAULT_TYPE
from asyncevo import initialize_member
from asyncevo import dispatch_work
from asyncevo import save
from asyncevo import load
from asyncevo import split_work


class ASAGa:
    """
    Adaptive Simulated Annealing (ASA) Genetic Algorithm.
    """
    dtype = DEFAULT_TYPE  # a single numpy type is used for all arrays

    def __init__(self,
                 initial_state: np.ndarray,
                 population_size: int,
                 scheduler: Scheduler,
                 global_seed: int,
                 sigma: float,
                 table_size: int = 20000000,
                 max_table_step: int = 5,
                 member_type=Member,
                 member_type_kwargs: Dict = None,
                 save_filename: Path = None,
                 save_every: int = None,
                 replacement_strategy: str = "crowding",
                 *args,
                 **kwargs):
        if member_type_kwargs is None:
            member_type_kwargs = {}

        if (cooling_factor < 0) or (cooling_factor > 1):
            raise AssertionError("Invalid input: Cooling factor must be"
                                 " between 0 and 1.")

        self._initial_state = initial_state.astype(dtype=ASAGa.dtype)
        self._population_size = population_size
        self._scheduler = scheduler
        self._global_seed = global_seed
        self._step = 0
        self._sigma0 = sigma
        self._sigma = sigma
        self._cooling_factor = cooling_factor
        self._annealing_start = annealing_start
        self._annealing_stop = annealing_stop
        self._table_size = table_size
        self._max_table_step = max_table_step
        self._save_filename = save_filename
        self._save_every = save_every
        self._from_file = kwargs.get("from_file", False)
        if replacement_strategy == "crowding":
            self._replacement = self._crowding
        elif replacement_strategy == "worst":
            self._replacement = self._worst

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
        self._member_parameters = {'initial_state': self._initial_state,
                                   'table_seed': self._table_seed,
                                   'table_size': self._table_size,
                                   'max_table_step': self._max_table_step}
        self._member_type_kwargs.update(self._member_parameters)

    def run(self, fitness_function: Callable[[np.ndarray], float],
            num_iterations: int,
            fitness_kwargs: Dict = None,
            take_member: bool = False):
        """
        Executes the genetic algorithm.
        :param fitness_function: a function that returns the fitness of a lineage.
        :param num_iterations: the number of steps to run.
        :param fitness_kwargs: any key word arguments for fitness_function.
        :param take_member: whether the fitness function requires the member to be
        provided or not (if not then expects an array) (default: False).
        """
        if fitness_kwargs is None:
            fitness_kwargs = {}

        if self._save_every is None:
            self._save_every = num_iterations

        # distribute members for each worker
        self._scheduler.wait_on_workers()
        num_workers = self._scheduler.num_workers()
        if num_workers == 0:
            raise ValueError("Error: there are no workers.")
        workers = self._scheduler.get_worker_names()
        if len(workers) == 0:
            raise ValueError("Error: there are no workers.")
        members = [self._scheduler.client.submit(initialize_member,
                                                 self._member_type,
                                                 self._member_type_kwargs,
                                                 workers=[worker])
                   for worker in workers]

        # skip if population is loaded from file
        if not self._from_file:
            # create initial batch
            initial_batch = [
                self._scheduler.client.submit(
                    dispatch_work,
                    fitness_function,
                    self._population[pop]['lineage'],
                    members[i], i, fitness_kwargs, take_member,
                    workers=[workers[i]])
                for i, pop_group in enumerate(
                    split_work(list(range(self._population_size)), num_workers))
                for pop in pop_group]

            # wait for initial batch to complete and fill initial population
            for completed_job in Scheduler.as_completed(initial_batch):
                fitness, lineage, index = completed_job.result()
                self._set_initial(lineage, fitness)

        # submit jobs to all workers or till num iterations is saturated
        jobs = []
        for index in range(min(num_iterations, len(members))):
            self._anneal()
            jobs.append(self._scheduler.client.submit(
                dispatch_work, fitness_function,
                self._mutation(self._selection()),
                members[index], index, fitness_kwargs,
                take_member=take_member,
                workers=[workers[index]]))
            self._step += 1

        # iterate ga until num_iterations reached
        working_batch = Scheduler.as_completed(jobs)
        for completed_job in working_batch:
            fitness, lineage, index = completed_job.result()
            self._replacement(lineage, fitness)
            self._update_fitness_history()
            if (self._save_filename is not None) and \
                    (self._step % self._save_every == 0):
                self.save_population(self._save_filename)

            if self._step < num_iterations:
                self._anneal()
                working_batch.add(self._scheduler.client.submit(
                    dispatch_work, fitness_function,
                    self._mutation(self._selection()),
                    members[index], index, fitness_kwargs,
                    take_member=take_member,
                    workers=[workers[index]]))
                self._step += 1

        if self._save_filename is not None:
            self.save_population(self._save_filename)