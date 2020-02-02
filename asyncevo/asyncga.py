__all__ = ['AsyncGa']


from asyncevo import Scheduler
from asyncevo import Distribution


class AsyncGa:
    """

    """
    def __init__(self, population_size: int,
                 initializing_distribution: Distribution,
                 scheduler: Scheduler,
                 global_seed: int,
                 sigma: float,
                 crossover_probability: float = 1.0,
                 cooling_factor: float = 1.0,
                 annealing_cut_off_step: int = -1):
        self._population_lineage = []

    def run(self, num_iterations: int):
        """

        :param num_iterations:
        :return:
        """
        pass
