__all__ = ['load', 'save', 'manhattan_distance']


import pickle
from pathlib import Path
from typing import Union
from typing import Any
import numpy as np


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
