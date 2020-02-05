__all__ = ['Lineage']


from collections.abc import Sequence


class Lineage(Sequence):
    """
    A Lineage represents the compressed form of the parameters. The sequence of
    seeds and sigmas are used to regenerate a member's parameters from some
    initial vector.
    """
    def __init__(self, seed: int = None, sigma: float = None):
        self._lineage = []
        if not (seed is None) and not (sigma is None):
            self._lineage.append({'seed': seed, 'sigma': sigma})

    def add_history(self, seed: int, sigma: float):
        self._lineage.append({'seed': seed, 'sigma': sigma})

    def __getitem__(self, i):
        return self._lineage[i]

    def __len__(self):
        return len(self._lineage)
