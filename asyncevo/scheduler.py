__all__ = ['Scheduler', 'scheduler']


class Scheduler:
    """

    """
    def __init__(self):
        """

        """
        from dask_mpi import initialize
        initialize()
        # Rank 0 is initialized with scheduler
        # Rank 1 will pass through and execute following code
        # Rank 2+ will execute workers

        from distributed import Client
        self._client = Client()

    @property
    def client(self):
        return self._client

    @client.setter
    def client(self, value):
        raise NotImplementedError

    def num_workers(self):
        """

        :return:
        """
        return len(self._client.scheduler_info()['workers'])


scheduler = Scheduler()
