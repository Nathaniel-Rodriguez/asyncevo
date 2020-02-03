__all__ = ['Lineage']


from dataclasses import dataclass
from typing import List


# lineage needs to be a tree of seeds for crossover
# at each node on tree it needs a corresponding sigma
@dataclass
class Lineage:
    """

    """
    seeds: List[int]  # change to tree
