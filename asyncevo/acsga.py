__all__ = ['ACSGa']


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
from asyncevo import ACSLineage
from asyncevo import DEFAULT_TYPE
from asyncevo import manhattan_distance
from asyncevo import initialize_member
from asyncevo import dispatch_work
from asyncevo import save
from asyncevo import load
from asyncevo import split_work
from asyncevo import AdaptiveCoolingSchedule


class ACSGa:
    """
    Adaptive Cooling Schedule (ACS) Genetic Algorithm.
    """
    dtype = DEFAULT_TYPE  # a single numpy type is used for all arrays

    def __init__(self,
                 initial_state: np.ndarray,
                 population_size: int,
                 scheduler: Scheduler,
                 global_seed: int,
                 sigma: float,
                 cooling_schedule: AdaptiveCoolingSchedule,
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

        self._initial_state = initial_state.astype(dtype=ACSGa.dtype)
        self._population_size = population_size
        self._scheduler = scheduler
        self._global_seed = global_seed
        self._step = 0
        self._sigma0 = sigma
        self._sigma = sigma
        self._cooling_schedule = cooling_schedule
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
        self._member_buffer1 = Member(**self._member_parameters)
        self._member_buffer2 = Member(**self._member_parameters)

    @classmethod
    def from_file(cls,
                  filename: Path,
                  scheduler: Scheduler,
                  global_seed: int,
                  sigma: float,
                  cooling_schedule: AdaptiveCoolingSchedule,
                  member_type=Member,
                  member_type_kwargs: Dict = None,
                  save_filename: Path = None,
                  save_every: int = None,
                  replacement_strategy: str = "crowding"):
        file_contents = load(filename)
        return cls(file_contents['initial_state'],
                   len(file_contents['population']),
                   scheduler,
                   global_seed,
                   sigma,
                   cooling_schedule,
                   file_contents['table_size'],
                   file_contents['max_table_step'],
                   member_type,
                   member_type_kwargs,
                   save_filename,
                   save_every,
                   replacement_strategy,
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
            self._anneal(fitness, lineage[-1]['temperature'])
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
        return [{'lineage': ACSLineage(self._make_seed(), self._sigma,
                                       self._cooling_schedule.temperature),
                 'fitness': None}
                for _ in range(self._population_size)]

    def _selection(self) -> ACSLineage:
        """
        Pick lineage at random from population based on ranked fitness
        :return: a lineage.
        """
        return self._population[self._rng.choices(self._fitness_sorted_indices(),
                                weights=self._selection_probabilities)[0]]['lineage']

    def _mutation(self, lineage: ACSLineage) -> ACSLineage:
        """
        Append a mutation operation onto a lineage in the form of a seed.
        It also appends the current sigma to the lineage.
        :param lineage: the lineage to mutate.
        :return: a new lineage.
        """
        mutant = deepcopy(lineage)
        mutant.add_history(self._make_seed(), self._sigma,
                           self._cooling_schedule.temperature)
        return mutant

    def _crowding(self, lineage: ACSLineage, fitness: float):
        """
        Crowding replacement. Given a lineage, express the member and find its
        manhattan distance to the parameters of other lineages in the
        population. If the given lineage is better than its closest neighbor,
        replace that neighbor in the population.
        :param lineage: a new lineage to check for replacement
        :param fitness: fitness of lineage
        """
        # We can skip replacement if the new lineage is weaker than all
        # other population members
        is_weakest = True
        for pop in self._population:
            if (pop['fitness'] is not None) and (pop['fitness'] < fitness):
                is_weakest = False
                break

        if is_weakest:
            return

        # Check the lineage's manhattan distance from all population members
        # and replace the closest one IF it is better.
        self._member_buffer1.appropriate_lineage(lineage)
        closest_member_index = None
        closest_distance = None
        for i, pop in enumerate(self._population):
            if pop['fitness'] is not None:
                self._member_buffer2.appropriate_lineage(pop['lineage'])
                distance = manhattan_distance(self._member_buffer1.parameters,
                                              self._member_buffer2.parameters)
                if closest_distance is None:
                    closest_distance = distance
                    closest_member_index = i
                elif distance < closest_distance:
                    closest_distance = distance
                    closest_member_index = i

        # Replacement does not take place if no evaluated closest member is found
        if (closest_member_index is not None) and \
                (self._population[closest_member_index]['fitness'] < fitness):
            self._population[closest_member_index] = {'lineage': lineage,
                                                      'fitness': fitness}

    def _worst(self, lineage: ACSLineage, fitness: float):
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

    def _anneal(self, sample_fitness, sample_temperature):
        """
        Applies annealing to sigma.
        """
        self._cooling_schedule.step(sample_temperature, -sample_fitness)

    def _set_initial(self, lineage: ACSLineage, fitness: float):
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
             'population': List[{'lineage': ACSLineage, 'fitness': float}],
             'initial_state': np.ndarray,
             'table_seed': int,
             'table_size': int,
             'max_table_step': int,
             'history': List[List['fitness]]
             'cooling_schedule': AdaptiveCoolingSchedule
             }

        :param filename: a file path
        """
        save({
            'population': self._population,
            'initial_state': self._initial_state,
            'table_seed': self._table_seed,
            'table_size': self._table_size,
            'max_table_step': self._max_table_step,
            'history': self._fitness_history,
            'cooling_schedule': self._cooling_schedule
        }, filename)