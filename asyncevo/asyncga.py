__all__ = ['AsyncGa']


from asyncevo import Scheduler
from asyncevo import Distribution
from asyncevo import Lineage
from typing import Dict


def initialize_member(member_cls, member_parameters: Dict):
    """

    :param member_cls:
    :param member_parameters:
    :return:
    """

    return member_cls(**member_parameters)


class AsyncGa:
    """

    """
    def __init__(self,
                 population_size: int,
                 initializing_distribution: Distribution,
                 scheduler: Scheduler,
                 global_seed: int,
                 sigma: float,
                 crossover_probability: float = 1.0,
                 cooling_factor: float = 1.0,
                 annealing_cut_off: int = -1):
        self._population_size = population_size
        self._initializing_distribution = initializing_distribution
        self._scheduler = scheduler
        self._global_seed = global_seed
        self._sigma = sigma
        self._crossover_probability = crossover_probability
        self._cooling_factor = cooling_factor
        self._annealing_cut_off = annealing_cut_off
        self._population_lineages = []
        self._population_fitnesses = []

    def run(self, num_iterations: int, lineages=None):
        """

        :param num_iterations:
        :param lineages:
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

    # probably do this:
    # pick lineage at random
    # lazy mutate -> new lineage
    # roll for crossover
    # if crossover
    #   pick second lineage at random
    #   lazy crossover -> new lineage (only 1 child)
    # dispatch new lineage
    def crossover(self,
                  seed: int,
                  first_parent_lineage: Lineage,
                  second_parent_lineage: Lineage) -> Lineage:
        """
        Single offspring crossover meant to replace the worst parent.
        :param seed:
        :param first_parent_lineage:
        :param second_parent_lineage:
        :return: lineage for offspring
        """
        pass

    def _selection(self):
        """
        Pick lineage at random
        :return:
        """

    def _replacement(self):
        """
        Do replace worst or replace parent(s)
        :return:
        """
        pass
    # need a replacement algorithm, with no duplicated
    # so either check == of two lineages
    # crowing is another option but requires check distance instantiated
    # parameters, and is more costly to
    # generate the agents locally to repeatedly to calculate Manhattan Distance
