__all__ = ['AsyncGa']


import numpy as np
from copy import deepcopy
from random import Random
from typing import Dict
from typing import List
from asyncevo import Scheduler
from asyncevo import Lineage
from asyncevo import Member


def initialize_member(member_cls, member_parameters: Dict):
    """

    :param member_cls:
    :param member_parameters:
    :return:
    """

    return member_cls(**member_parameters)


class AsyncGa:
    """
    An asynchronous steady-state genetic algorithm that uses random selection
    and crowding replacement. An exponential annealing schedule is used to
    change the mutation size over time.
    """
    dtype = np.float32  # use a single type throughout

    def __init__(self,
                 initial_state: np.ndarray,
                 population_size: int,
                 scheduler: Scheduler,
                 global_seed: int,
                 sigma: float,
                 cooling_factor: float = 1.0,
                 annealing_start: int = 0,
                 annealing_stop: int = -1,
                 table_size: int = 20000000,
                 max_table_step: int = 5):
        """
        :param initial_state: a numpy array with the initial parameter guess.
        :param population_size: the desired size of the evolutionary population.
        :param scheduler: a scheduler.
        :param global_seed: a seed that will be used for whole simulation.
        :param sigma: initial mutation size.
        :param cooling_factor: factor for simulated annealing. Must be bound
            between [0,1]. A factor of 1.0 means no annealing takes place.
        :param annealing_start: step at which to begin annealing (default 0).
        :param annealing_stop: step at which to stop annealing (default -1). A
            value of -1 means that annealing will not end.
        :param table_size: the size of the random number table for members.
        :param max_table_step: the maximum random stride for table slices
        """
        self._initial_state = initial_state.astype(dtype=AsyncGa.dtype)
        self._population_size = population_size
        self._scheduler = scheduler
        self._global_seed = global_seed
        self._step = 0
        self._sigma = sigma
        self._cooling_factor = cooling_factor
        self._annealing_start = annealing_start
        self._annealing_stop = annealing_stop
        self._table_size = table_size
        self._max_table_step = max_table_step
        self._population = self._initialize()
        self._rng = Random(self._global_seed)
        self._member_buffer1 = Member()
        self._member_buffer2 = Member()

    def run(self, num_iterations: int):
        """
        Executes the genetic algorithm.
        :param num_iterations: the number of steps to run.
        :return:
        """

        # scatter member initializer
        # generate lineages, if no lineages, else use available
        # do initial map of fitness function with lineages
        # define steps
        # for loop over future as_completed
            # get result
            # ???
            # check number of iterations, if not finished
                # submit new work
                # increment steps
        # update lineages -> save to file

    def _make_seed(self) -> int:
        """

        :return:
        """
        return self._rng.randint(0, 1000000)

    def _initialize(self) -> List[Dict]:
        """
        Generates the initial population of lineages with None fitness values.
        :return: a list of dictionaries with keys 'lineage' and 'fitness'
        """
        return [{'lineage': Lineage([self._make_seed()], [self._sigma])}
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
        """
        pass

    def _anneal(self):
        """
        Applies annealing to sigma.
        """
        self._sigma = 0  # calc new sigma here, based on self._step

    # need a replacement algorithm, with no duplicated
    # so either check == of two lineages

    # crowding is another option but requires check distance instantiated
    # parameters, and is more costly to
    # generate the agents locally to repeatedly to calculate Manhattan Distance
    # HOWEVER, the master isn't really doing a whole lot after submissions
    # it could keep a PxP table w/ manhattan distances between members
    # that it generates. Whenever a new member is introduced, its distance
    # would need to be calculated against all other members. So all the other
    # members would need to be recreated. But this is done anyway in ES
    # with crowding you still only replace if it has better fitness
    # in crowding you replace the member closest to you in space if
    # your fitness is higher
