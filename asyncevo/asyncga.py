__all__ = ['AsyncGa']


from asyncevo import Scheduler
from asyncevo import Distribution
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
        self._population_lineage = []

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
