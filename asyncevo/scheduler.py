__all__ = ['Scheduler']


from typing import Dict


def initialize_member(member_cls, member_parameters: Dict):
    """

    :param member_cls:
    :param member_parameters:
    :return:
    """

    return member_cls(**member_parameters)


class Scheduler:
    """

    """
    def __init__(self, num_workers: int,
                 num_threads_per_worker: int = 1):
        """

        """
        pass

    def initialize_population(self, member_cls, member_parameters):
        """

        :param member_cls:
        :param member_parameters:
        :return:
        """
