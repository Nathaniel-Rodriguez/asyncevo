__all__ = ['Member', 'RandNumTable']


from numpy import np
from asyncevo.distribution import Distribution
from asyncevo.lineage import Lineage


# we don't need two members for crossover!
# 1 member is enough.
# we use the lineage -> integers RV for table
# initializing dist is a problem
# we have to go through add/subtract mutations for lineages on single member
# lineage just says what mutation should be.. and they are all additive, so
# order applied doesn't matter (and multiplicative, but same idea, it just
# scales the addition)
class Member:
    """

    """
    def __init__(self, initializing_distribution: Distribution):
        """

        """
        self._initializing_distribution = initializing_distribution

    def mutate(self, seed):
        pass


class RandNumTable:
    """

    """
    def __init__(self, table_size: int, seed: int):
        """

        :param table_size:
        :param seed:
        """
        self._rng = np.random.RandomState(seed)
        self._table = self._rng.randn(table_size)
