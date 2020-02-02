__all__ = ['Member', 'RandNumTable']


from numpy import np
from asyncevo.distribution import Distribution


class Member:
    """

    """
    def __init__(self, initializing_distribution: Distribution):
        """

        """
        self._initializing_distribution = initializing_distribution


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
