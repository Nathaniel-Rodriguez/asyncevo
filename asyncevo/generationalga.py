__all__ = ['GenerationalGa']


import numpy as np
from math import inf
from copy import deepcopy
from pathlib import Path
from random import Random
from typing import Dict
from typing import Callable
from typing import List
from asyncevo import Scheduler
from asyncevo import Lineage
from asyncevo import Member
from asyncevo import split_work
from asyncevo import save
from asyncevo import load
from asyncevo import DEFAULT_TYPE
from asyncevo import initialize_member
from asyncevo import dispatch_work


class GenerationalGa:
    """
    A generational genetic algorithm that uses rank fitness selection and random
    replacement. An exponential annealing schedule is used to change the
    mutation size over time.
    """
    dtype = DEFAULT_TYPE  # a single numpy type is used for all arrays

    def __init__(self,
                 initial_state: np.ndarray,
                 population_size: int,
                 scheduler: Scheduler,
                 global_seed: int,
                 sigma: float,
                 cooling_factor: float = 1.0,
                 annealing_start: int = 0,
                 annealing_stop: int = inf,
                 table_size: int = 20000000,
                 max_table_step: int = 5,
                 member_type=Member,
                 member_type_kwargs: Dict = None,
                 save_filename: Path = None,
                 save_every: int = None,
                 *args,
                 **kwargs):
        """
        :param initial_state: a numpy array with the initial parameter guess.
        :param population_size: the desired size of the evolutionary population.
        :param scheduler: a scheduler.
        :param global_seed: a seed that will be used for whole simulation.
        :param sigma: initial mutation size.
        :param cooling_factor: factor for simulated annealing. Must be bound
            between [0,1]. A factor of 1.0 means no annealing takes place.
        :param annealing_start: step at which to begin annealing (default 0).
        :param annealing_stop: step at which to stop annealing (default inf). A
            value of -1 means that annealing will not end.
        :param table_size: the size of the random number table for members.
        :param max_table_step: the maximum random stride for table slices
        :param member_type: specifies what type to use for a member, should
            use Member or a subclass of Member (default to Member). The subclass
            must be able to consume and forward all of Member's arguments to it.
            One reason to subclass Member is to keep additional information
            stored on workers over the duration of the run.
        :param member_type_kwargs: additional keyword arguments not related
        to the base Member arguments.
        :param save_filename: a filename or path to save the output to.
        :param save_every: save population every X number of steps
        (default saves only at end).
        """
        if member_type_kwargs is None:
            member_type_kwargs = {}

        if (cooling_factor < 0) or (cooling_factor > 1):
            raise AssertionError("Invalid input: Cooling factor must be"
                                 " between 0 and 1.")

        self._initial_state = initial_state.astype(dtype=GenerationalGa.dtype)
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
        self._member_buffer1 = Member(**self._member_parameters)
        self._member_buffer2 = Member(**self._member_parameters)

    @classmethod
    def from_file(cls,
                  filename: Path,
                  scheduler: Scheduler,
                  global_seed: int,
                  sigma: float,
                  cooling_factor: float = 1.0,
                  annealing_start: int = 0,
                  annealing_stop: int = inf,
                  member_type=Member,
                  member_type_kwargs: Dict = None,
                  save_filename: Path = None,
                  save_every: int = None):
        file_contents = load(filename)
        return cls(file_contents['initial_state'],
                   len(file_contents['population']),
                   scheduler,
                   global_seed,
                   sigma,
                   cooling_factor,
                   annealing_start,
                   annealing_stop,
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

        # submit jobs to all workers
        for index in range(num_iterations):
            self._step += 1
            self._anneal()
            self._selection()
            self._mutation()
            generation_batch = [
                self._scheduler.client.submit(
                    dispatch_work, fitness_function,
                    self._population[pop]['lineage'],
                    members[index], index, fitness_kwargs,
                    take_member=take_member,
                    workers=[workers[index]])
                for i, pop_group in enumerate(
                    split_work(list(range(self._population_size)), num_workers))
                for pop in pop_group
            ]

            Scheduler.wait(generation_batch)
            for i, job in enumerate(generation_batch):
                fitness, lineage, index = job.result()
                self._update_pop(i, lineage, fitness)

            self._update_fitness_history()
            if (self._save_filename is not None) and \
                    (self._step % self._save_every == 0):
                self.save_population(self._save_filename)

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
        return [{'lineage': Lineage(self._make_seed(), self._sigma),
                 'fitness': None}
                for _ in range(self._population_size)]

    def _selection(self):
        """
        Pick lineages at random from population based on ranked fitness.
        """
        chosen = self._rng.choices(self._fitness_sorted_indices(),
                                   weights=self._selection_probabilities,
                                   k=self._population_size)
        self._population = [self._population[index] for index in chosen]

    def _mutation(self):
        """
        Append a mutation operation lineages in the population.
        """
        for pop in self._population:
            pop['lineage'].add_history(self._make_seed(), self._sigma)

    def _update_pop(self, pop_index, lineage, fitness):
        """
        Update the population with a new member.
        :param pop_index: the population member index to replace
        :param lineage: the replacement lineage
        :param fitness: the new fitness
        """
        self._population[pop_index]['lineage'] = lineage
        self._population[pop_index]['fitness'] = fitness

    def _anneal(self):
        """
        Applies annealing to sigma.
        """
        if (self._step > self._annealing_start) \
           and (self._step < self._annealing_stop):
            self._sigma = self._sigma0 * (self._cooling_factor **
                                          (self._step - self._annealing_start))

    def _set_initial(self, lineage: Lineage, fitness: float):
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
            'table_seed': self._table_seed,
            'table_size': self._table_size,
            'max_table_step': self._max_table_step,
            'history': self._fitness_history
        }, filename)
