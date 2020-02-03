__all__ = ['load', 'save']


import pickle
from pathlib import Path
from typing import Union
from typing import Any


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
