from asyncevo.scheduler import Scheduler


# A global scheduler that gets created when scheduler is imported.
# scheduler should be imported first to ensure user code is only executed
# on the master rank (rank=1)
mpi_scheduler = Scheduler()
