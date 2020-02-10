from typing import List


__all__ = ['Scheduler']


class Scheduler:
    """
    A wrapper around dask-mpi.
    """
    def __init__(self):
        """
        Creating a scheduler initializes dask MPI
        """

        from dask_mpi import initialize
        initialize()
        # Rank 0 is initialized with scheduler
        # Rank 1 will pass through and execute following code
        # Rank 2+ will execute workers

        from distributed import Client
        self._client = Client()  # TODO specify memory

    @property
    def client(self):
        return self._client

    @client.setter
    def client(self, value):
        raise NotImplementedError

    def num_workers(self) -> int:
        """
        :return: the number of known workers.
        """
        return len(self._client.scheduler_info()['workers'])

    def get_worker_names(self) -> List[str]:
        """
        :return: return list of worker names
        """
        return list(self._client.scheduler_info()['workers'].keys())

    @staticmethod
    def as_completed(*args, **kwargs):
        from distributed import as_completed
        return as_completed(*args, **kwargs)
