__all__ = ['Scheduler']


from typing import List
from typing import Dict
from distributed import Client
from distributed import as_completed


class Scheduler:
    """
    A wrapper around dask-mpi.
    """
    def __init__(self, initialization_args: Dict = None,
                 client_args: Dict = None):
        """
        Creating a scheduler initializes dask MPI
        :param initialization_args: dictionary of keyword arguments for dask-mpi
        initializer.
        :param client_args: dictionary of keyword arguments for dask client.


        dask-mpi arguments
        ------------------
        interface:str
        Network interface like ‘eth0’ or ‘ib0’

        nthreads:int
        Number of threads per worker

        local_directory:str
        Directory to place worker files

        memory_limit:int, float, or ‘auto’
        Number of bytes before spilling data to disk. This can be an integer (nbytes), float (fraction of total memory), or ‘auto’.

        nanny:bool
        Start workers in nanny process for management

        bokeh:bool
        Enable Bokeh visual diagnostics

        bokeh_port:int
        Bokeh port for visual diagnostics

        bokeh_prefix:str
        Prefix for the bokeh app

        bokeh_worker_port:int
        Worker’s Bokeh port for visual diagnostics


        dask client arguments
        ---------------------
        address: string, or Cluster
        This can be the address of a Scheduler server like a string
        '127.0.0.1:8786' or a cluster object like LocalCluster()

        timeout: int
        Timeout duration for initial connection to the scheduler

        set_as_default: bool (True)
        Claim this scheduler as the global dask scheduler

        scheduler_file: string (optional)
        Path to a file with scheduler information if available

        security: Security or bool, optional
        Optional security information. If creating a local cluster can also
        pass in True, in which case temporary self-signed credentials will
        be created automatically.

        asynchronous: bool (False by default)
        Set to True if using this client within async/await functions or within
        Tornado gen.coroutines. Otherwise this should remain False for normal use.

        name: string (optional)
        Gives the client a name that will be included in logs generated on the
        scheduler for matters relating to this client

        direct_to_workers: bool (optional)
        Whether or not to connect directly to the workers, or to ask the
        scheduler to serve as intermediary.

        heartbeat_interval: int
        Time in milliseconds between heartbeats to scheduler
        """
        if initialization_args is None:
            initialization_args = {}

        if client_args is None:
            client_args = {}

        from dask_mpi import initialize
        initialize(**initialization_args)
        # Rank 0 is initialized with scheduler
        # Rank 1 will pass through and execute following code
        # Rank 2+ will execute workers
        self._client = Client(**client_args)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._client.shutdown()

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
        return as_completed(*args, **kwargs)
