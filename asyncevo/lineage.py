__all__ = ['Lineage', 'GaLineage']


from dataclasses import dataclass
from typing import List


@dataclass
class Lineage:
    """

    """
    seeds: List[int]


@dataclass
class GaLineage(Lineage):
    """

    """
    sigmas: List[float]
    crossover_steps: List[int]
