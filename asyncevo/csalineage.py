__all__ = ['CSALineage']


from collections.abc import Sequence


class CSALineage(Sequence):
    """
    A CSALineage represents the compressed form of the parameters used to
    regenerate a member's parameters from an initial vector. This lineage
    contains the seeds and fitness of all past members of the population.
    """
    def __init__(self, seed: int = None, fitness: float = None):
        self._lineage = []
        if not (seed is None) and not (fitness is None):
            self._lineage.append({'seed': seed, 'fitness': fitness})

    def add_history(self, seed: int, fitness: float):
        self._lineage.append({'seed': seed, 'fitness': fitness})

    def __getitem__(self, i):
        return self._lineage[i]

    def __len__(self):
        return len(self._lineage)

    def __eq__(self, other):
        if len(other) != len(self):
            return False

        for i in range(len(self)):
            if not (self[i]['seed'] == other[i]['seed']
                    and self[i]['fitness'] == other[i]['fitness']):
                return False

        return True
