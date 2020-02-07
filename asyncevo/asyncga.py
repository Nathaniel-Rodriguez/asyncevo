__all__ = ['AsyncGa']


import numpy as np
from math import inf
from copy import deepcopy
from random import Random
from typing import Dict
from typing import List
from typing import Tuple
from typing import Callable
from asyncevo import Scheduler
from asyncevo import Lineage
from asyncevo import Member
from asyncevo import manhattan_distance


def initialize_member(member_type, member_parameters: Dict) -> Member:
    """
    A wrapper function for initializing members on workers.
    :param member_type: Member or a subclass of Member.
    :param member_parameters: parameters initializing the member.
    :return: A member
    """
    return member_type(**member_parameters)


def dispatch_work(fitness_function: Callable[[np.ndarray, ...], float],
                  lineage: Lineage,
                  member: Member,
                  fitness_kwargs=None) -> Tuple[float, Lineage]:
    """
    Sends out work to a member and returns a tuple of the fitness and its
    associated lineage. The lineage is returned so that dispatch_work can
    be used in an asynchronous scenario where
    :param fitness_function: a callable object that takes member parameters
    numpy array and returns a fitness value.
    :param lineage: the lineage for the member to use to generate parameters.
    :param member: an initialized member.
    :param fitness_kwargs: additional arguments for the fitness function
    :return: fitness
    """
    if fitness_kwargs is None:
        fitness_kwargs = {}

    member.appropriate_lineage(lineage)
    return fitness_function(member.parameters, **fitness_kwargs), lineage


class AsyncGa:
    """
    An asynchronous steady-state genetic algorithm that uses random selection
    and crowding replacement. An exponential annealing schedule is used to
    change the mutation size over time.
    """
    dtype = np.float32  # a single numpy type is used for all arrays

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
                 member_type=Member):
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
        """
        if (cooling_factor < 0) or (cooling_factor > 1):
            raise AssertionError("Invalid input: Cooling factor must be"
                                 " between 0 and 1.")

        self._initial_state = initial_state.astype(dtype=AsyncGa.dtype)
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
        self._population = self._initialize()
        self._rng = Random(self._global_seed)
        self._table_seed = self._make_seed()
        self._member_type = member_type
        self._member_parameters = {'initial_state': self._initial_state,
                                   'table_seed': self._table_seed,
                                   'table_size': self._table_size,
                                   'max_table_step': self._max_table_step}
        self._member_buffer1 = Member(**self._member_parameters)
        self._member_buffer2 = Member(**self._member_parameters)

    def run(self, fitness_function: Callable[[np.ndarray, ...], float],
            num_iterations: int,
            fitness_kwargs=None):
        """
        Executes the genetic algorithm.
        :param fitness_function: a function that returns the fitness of a lineage
        :param num_iterations: the number of steps to run.
        :param fitness_kwargs: any key word arguments for fitness_function
        """
        if fitness_kwargs is None:
            fitness_kwargs = {}

        # generate members on each worker
        members = self._scheduler.client.map(initialize_member,
            [self._member_type for _ in self._scheduler.num_workers()],
            member_parameters=self._member_parameters)

        # create initial batch
        initial_batch = []
        for i, member in enumerate(members):
            initial_batch.append(self._scheduler.client.submit(
                dispatch_work, fitness_function, self._population[i]['lineage'],
                member, **fitness_kwargs))
        # TODO: need to put in rest of population

        working_batch = Scheduler.as_completed(initial_batch)
        for result in working_batch:
            self._replacement(result[1], result[0])
            if self._step < num_iterations:
                self._anneal()
                new_lineage = self._mutation(self._selection())
                working_batch.add(self._scheduler.client.submit(
                    dispatch_work, fitness_function, new_lineage,
                    member, **fitness_kwargs))  # TODO: figure out which member to give
                self._step += 1
                
        # TODO: update lineages -> save to file

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

    def _selection(self) -> Lineage:
        """
        Pick lineage at random from population.
        :return: a lineage.
        """
        return self._rng.choice(self._population)['lineage']

    def _mutation(self, lineage: Lineage) -> Lineage:
        """
        Append a mutation operation onto a lineage in the form of a seed.
        It also appends the current sigma to the lineage.
        :param lineage: the lineage to mutate.
        :return: a new lineage.
        """
        mutant = deepcopy(lineage)
        mutant.add_history(self._make_seed(), self._sigma)
        return mutant

    def _replacement(self, lineage: Lineage, fitness: float):
        """
        Crowding replacement. Given a lineage, express the member and find its
        manhattan distance to the parameters of other lineages in the
        population. If the given lineage is better than its closest neighbor,
        replace that neighbor in the population.
        :param lineage: a new lineage to check for replacement
        :param fitness: fitness of lineage
        """

        is_weakest = True
        for pop in self._population:
            if (pop['fitness'] is not None) and (pop['fitness'] > fitness):
                is_weakest = False

        if is_weakest:
            return

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

        if self._population[closest_member_index]['fitess'] < fitness:
            self._population[closest_member_index] = {'lineage': lineage,
                                                      'fitness': fitness}

    def _anneal(self):
        """
        Applies annealing to sigma.
        """
        if (self._step > self._annealing_start) \
           and (self._step < self._annealing_stop):
            self._sigma = self._sigma0 * (self._cooling_factor ** self._step)
