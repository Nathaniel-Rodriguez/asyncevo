__all__ = ['ACSLineage']


from collections.abc import Sequence


class ACSLineage(Sequence):
    """
    An ACSLineage represents the compressed form of the parameters.
    The sequence of seeds, sigmas, and temperatures are used to regenerate
    a member's parameters from some initial vector.
    """
    def __init__(self, seed: int = None, sigma: float = None,
                 temperature: float = None):
        self._lineage = []
        if not (seed is None) and not (sigma is None)\
                and not (temperature is None):
            self._lineage.append({'seed': seed, 'sigma': sigma,
                                  'temperature': temperature})

    def add_history(self, seed: int, sigma: float, temperature: float):
        self._lineage.append({'seed': seed, 'sigma': sigma,
                              'temperature': temperature})

    def __getitem__(self, i):
        return self._lineage[i]

    def __len__(self):
        return len(self._lineage)

    def __eq__(self, other):
        if len(other) != len(self):
            return False

        for i in range(len(self)):
            if not (self[i]['seed'] == other[i]['seed']
                    and self[i]['sigma'] == other[i]['sigma']
                    and self[i]['temperature'] == other[i]['temperature']):
                return False

        return True
