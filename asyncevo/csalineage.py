__all__ = ['CSALineage']


from typing import List
from collections import namedtuple


EvoPathMarker = namedtuple('EvoPathMarker', ['seed', 'fitness'])


class CSALineage:
    """
    A CSALineage represents the compressed form of the parameters used to
    regenerate a member's parameters from an initial vector. This lineage
    contains the seeds and fitness of all past members of the population.
    """
    def __init__(self, seed: int = None):
        self._lineage = []
        self._path_lineage = []
        if seed is not None:
            self._lineage.append(seed)

    def add_lineage_history(self, seed: int):
        self._lineage.append(seed)

    def add_path_history(self, population_markers: List[EvoPathMarker]):
        self._path_lineage.append(population_markers)

    @property
    def lineage(self):
        return self._lineage

    @lineage.setter
    def lineage(self, value):
        raise NotImplementedError

    @property
    def path(self):
        return self._path_lineage

    @path.setter
    def path(self, value):
        raise NotImplementedError

    def __eq__(self, other):
        if len(other.lineage) != len(self.lineage):
            return False

        if len(other.path) != len(self.path):
            return False

        for i in range(len(self.lineage)):
            if other.lineage[i] != self.lineage[i]:
                return False

        for i in range(len(self.path)):
            if other.path[i] != self.path[i]:
                return False

        return True
