__all__ = ['initialize_member', 'dispatch_work']


from typing import Callable
from typing import Union
from typing import Tuple
from typing import Dict
from typing import Any
from asyncevo.basemember import BaseMember
import numpy as np


def initialize_member(member_type, member_parameters: Dict) -> BaseMember:
    """
    A wrapper function for initializing members on workers.
    :param member_type: Member or a subclass of Member.
    :param member_parameters: parameters initializing the member.
    :return: A member
    """
    return member_type(**member_parameters)


def dispatch_work(fitness_function: Union[Callable[[BaseMember], float],
                                          Callable[[np.ndarray], float]],
                  lineage: Any,
                  member: BaseMember,
                  member_id: int,
                  fitness_kwargs: Dict = None,
                  take_member: bool = False) -> Tuple[float, Any, int]:
    """
    Sends out work to a member and returns a tuple of the fitness and its
    associated lineage. The lineage is returned so that dispatch_work can
    be used in an asynchronous scenario where
    :param fitness_function: a callable object that takes member parameters
    numpy array and returns a fitness value. It can also be a callable
    that takes as an argument a Member or subclass of Member and returns the
    fitness.
    :param lineage: the lineage for the member to use to generate parameters.
    :param member: an initialized member.
    :param member_id: id of the member.
    :param fitness_kwargs: additional arguments for the fitness function
    :param take_member: whether the fitness function requires the member to be
    provided or not (if not then expects an array) (default: False).
    :return: fitness, lineage, member id, and is_initial
    """
    if fitness_kwargs is None:
        fitness_kwargs = {}

    member.appropriate_lineage(lineage)
    if take_member:
        return fitness_function(member, **fitness_kwargs),\
               lineage, member_id

    else:
        return fitness_function(member.parameters, **fitness_kwargs),\
               lineage, member_id
