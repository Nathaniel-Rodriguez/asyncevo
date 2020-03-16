__all__ = ['CSALineage']


from collections.abc import Sequence


# add own path seperate from lineage??
# problem with asynchrony... our parents maybe dead or
# we may have an old step that gave rise to us.
# so what we really need is to know every pop that happened up to us
# and our branch from it.
# at any point in time there will be many evo paths
class CSALineage(Sequence):
    """
    A CSALineage represents the compressed form of the parameters used to
    regenerate a member's parameters from an initial vector. This lineage
    contains the seeds and fitness of all past members of the population.
    """
    def __init__(self, seed: int = None):
        self._own_seed = seed
        self._lineage = []

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
