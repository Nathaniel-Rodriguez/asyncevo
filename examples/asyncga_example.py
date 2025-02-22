# run this example with:
# <mpi command> -n <number cores> python asyncga_example.py

# Initialize the MPI process
# all code following this code will be executed on rank 1
# rank 0 is dedicated to the scheduler
# ranks 2+ are dedicated to workers
from asyncevo import Scheduler
from asyncevo import AsyncGa
from asyncevo import Member
import numpy as np
from math import inf
from pathlib import Path
from time import sleep


# Below are available fitness functions
def elli(x):
    """ellipsoid-like test fitness function"""
    n = len(x)
    aratio = 1e3
    return -sum(x[i]**2 * aratio**(2.*i/(n-1)) for i in range(n))


def sphere(x):
    """sphere-like, ``sum(x**2)``, test fitness function"""
    return -sum(x[i]**2 for i in range(len(x)))


def rosenbrock(x):
    """Rosenbrock test fitness function"""
    n = len(x)
    if n < 2:
        raise ValueError('dimension must be greater than one')
    return -sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
                for i in range(n-1))


def rosenbrock2d(x):
    """
    Best at f(1,1)=0
    """
    return -((1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2)


def rest(x):
    """sleeps for a time and then returns sphere."""
    sleep(np.random.randint(1, 3))
    return sphere(x)


def member_example(member):
    return sphere(member.parameters)


def main():
    # make the scheduler first
    with Scheduler({'nanny': True, 'interface': 'lo'}) as mpi_scheduler:
        # create and run GA
        # ga = AsyncGa(initial_state=np.array([0.4, 0.3, -0.25, 0.01]),
        #              population_size=5,
        #              scheduler=mpi_scheduler,
        #              global_seed=96879,
        #              sigma=5.0,
        #              cooling_factor=0.996,
        #              annealing_start=0,
        #              annealing_stop=inf,
        #              table_size=20000,
        #              max_table_step=1,
        #              member_type=Member,
        #              save_filename=Path("test.asyncga"),
        #              save_every=1000)
        # ga.run(sphere, 350, take_member=False)

        ga = AsyncGa(initial_state=np.array([0.4, 0.3, -0.25, 0.01]),
                     population_size=5,
                     scheduler=mpi_scheduler,
                     global_seed=96879,
                     sigma=5.0,
                     cooling_factor=0.996,
                     annealing_start=0,
                     annealing_stop=inf,
                     table_size=20000,
                     max_table_step=1,
                     member_type=Member,
                     save_filename=Path("test.asyncga"),
                     save_every=1000)
        ga.run(rosenbrock, 350, take_member=False)

        # load pop from file and continue
        # ga = AsyncGa.from_file("test.asyncga",
        #                        scheduler=mpi_scheduler,
        #                        global_seed=432,
        #                        sigma=0.1,
        #                        cooling_factor=1.0,
        #                        save_filename="test.asyncga")
        # ga.run(sphere, 50, take_member=False)


if __name__ == "__main__":
    main()
