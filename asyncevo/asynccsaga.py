__all__ = ['AsyncCSAGa']

import numpy as np
from typing import Dict
from pathlib import Path
from random import Random
from math import inf
from copy import deepcopy
from asyncevo import Scheduler
from asyncevo import CSALineage
from asyncevo import CSAMember
from asyncevo import split_work
from asyncevo import save
from asyncevo import load
from asyncevo import DEFAULT_TYPE
from asyncevo import initialize_member
from asyncevo import dispatch_work


class AsyncCSAGa:
    """
    An asynchronous stead-state genetic algorithm that uses random selection
    and worst replacement. Additionally, it uses cumulative step-size
    adaptation (CSA) which uses parameter-wise evolutionary paths to adapt
    the step-size of the mutations over time.
    """
    dtype = DEFAULT_TYPE

    def __init__(self,
                 initial_state: np.ndarray,
                 initial_sigma: np.ndarray,
                 population_size: int,
                 scheduler: Scheduler,
                 global_seed: int,
                 path_memory: float,
                 adaptation_speed: float,
                 adaptation_precision: float,
                 table_size: int = 20000000,
                 max_table_step: int = 5,
                 member_type=CSAMember,
                 member_type_kwargs: Dict = None,
                 save_filename: Path = None,
                 save_every: int = None,
                 *args,
                 **kwargs):
        pass
