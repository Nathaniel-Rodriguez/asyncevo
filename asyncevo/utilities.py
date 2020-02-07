__all__ = ['load', 'save', 'manhattan_distance', 'split_work']


import pickle
from pathlib import Path
from typing import Union
from typing import Any
from typing import List
from typing import Generator
import numpy as np


def split_work(work: List[Any], num_batches: int) -> Generator[List[Any], None, None]:
    """
    Divides a list of work into evenly (as best as possible) sub batches.
    :param work: a list of work
    :param num_batches: the number of sublists to split it into
    :return: a generator that return lists
    """
    for i in range(0, len(work), num_batches):
        yield work[i:i + num_batches]


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Uses the second point 'b' to hold temporary values in calculation.
    Point 'a' is not modified.
    :param a: a numpy array
    :param b: a numpy array
    :return: the manhattan distance between two arrays
    """
    b -= a
    np.fabs(b, out=b)
    return b.sum()


def load(filename: Union[Path, str]) -> Any:
    """

    :param filename:
    :return:
    """
    pickled_obj_file = open(filename, 'rb')
    obj = pickle.load(pickled_obj_file)
    pickled_obj_file.close()

    return obj


def save(data: Any,
         filename: Union[Path, str],
         protocol=pickle.DEFAULT_PROTOCOL):
    pickled_obj_file = open(filename, 'wb')
    pickle.dump(data, pickled_obj_file, protocol=protocol)
    pickled_obj_file.close()
