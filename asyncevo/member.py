from numpy import np


class Member:
    """

    """
    def __init__(self, member_size):
        """

        """
        pass


class RandNumTableMixin:
    """

    """
    def __init__(self, table_size, seed):
        """

        :param table_size:
        :param seed:
        """
        self._rng = np.random.RandomState(seed)
        self._table = self._rng.randn(table_size)
