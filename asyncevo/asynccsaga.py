__all__ = ['AsyncCSAGa']

import numpy as np
from typing import Dict
from typing import Callable
from typing import List
from pathlib import Path
from random import Random
from math import inf
from math import sqrt
from copy import deepcopy
from asyncevo import Scheduler
from asyncevo import CSALineage
from asyncevo import EvoPathMarker
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
        self._member_parameters = {'initial_state': self._initial_state,
                                   'initial_sigma': self._initial_sigma,
                                   'table_seed': self._table_seed,
                                   'table_size': self._table_size,
                                   'max_table_step': self._max_table_step,
                                   'path_memory': self._path_memory,
                                   'adaptation_speed': self._adaptation_speed,
                                   'adaptation_precision': self._adaptation_precision,
                                   'population_size': self._population_size,
                                   'dtype': AsyncCSAGa.dtype}
        self._member_type_kwargs.update(self._member_parameters)

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
                working_batch.add(self._scheduler.client.submit(
                    dispatch_work, fitness_function,
                    self._mutation(self._selection()),
                    members[index], index, fitness_kwargs,
                    take_member=take_member,
                    workers=[workers[index]]))
                self._step += 1

        if self._save_filename is not None:
            self.save_population(self._save_filename)

    def _fitness_sorted_indices(self) -> np.ndarray:
        """
        :return: an array of pop indices sorted by greatest fitness to least
        """
        return np.flip(np.argsort([pop['fitness']
                                   if pop['fitness'] is not None else -np.inf
                                   for pop in self._population]))

    def _make_seed(self) -> int:
        """
        Generates a new seed from the master rng.
        :return: a new seed.
        """
        return self._rng.randint(0, 1000000)

    def _initialize(self) -> List[Dict]:
        """
        Generates the initial population of lineages with None fitness values.
        :return: a list of dictionaries with keys 'lineage' and 'fitness'
        """
        return [{'lineage': CSALineage(self._make_seed()),
                 'fitness': None}
                for _ in range(self._population_size)]

    def _selection(self) -> CSALineage:
        """
        Pick lineage at random from population based on ranked fitness
        :return: a lineage.
        """
        return self._population[self._rng.choices(self._fitness_sorted_indices(),
                                weights=self._selection_probabilities)[0]]['lineage']

    def _mutation(self, lineage: CSALineage) -> CSALineage:
        """
        Append a mutation operation onto a lineage in the form of a seed.
        It also appends the current sigma to the lineage.
        :param lineage: the lineage to mutate.
        :return: a new lineage.
        """
        mutant = deepcopy(lineage)
        mutant.add_lineage_history(self._make_seed())
        mutant.add_path_history([EvoPathMarker(pop['seed'], pop['fitness'])
                                 for pop in self._population])
        return mutant

    def _replacement(self, lineage: CSALineage, fitness: float):
        """
        Worst replacement. Given a lineage, if the given lineage is better than
        the worst population neighbor, replace that member in the population.
        :param lineage: a new lineage to check for replacement
        :param fitness: fitness of lineage
        """
        # find lowest fitness pop and then replace
        weakest_member = None
        lowest_fitness = inf
        for i, pop in enumerate(self._population):
            if (pop['fitness'] is not None) and (pop['fitness'] < lowest_fitness):
                weakest_member = i
                lowest_fitness = pop['fitness']

        if fitness > lowest_fitness and weakest_member is not None:
            self._population[weakest_member]['lineage'] = lineage
            self._population[weakest_member]['fitness'] = fitness

    def _set_initial(self, lineage: CSALineage, fitness: float):
        """
        Helps initialize the population with its first members.
        :param lineage: a lineage
        :param fitness: a fitness
        """
        is_set = False
        for pop in self._population:
            if pop['lineage'] == lineage:
                pop['fitness'] = fitness
                is_set = True

        if not is_set:
            raise ValueError("Lineage from initial batch not set.")

    def _update_fitness_history(self):
        """
        Updates the internal record of fitness for the population.
        """
        self._fitness_history.append([pop['fitness']
                                      for pop in self._population
                                      if pop['fitness'] is not None])

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
