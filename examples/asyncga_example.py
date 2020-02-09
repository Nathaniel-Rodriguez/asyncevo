# Initialize the MPI process
# all code following this code will be executed on rank 1
# rank 0 is dedicated to the scheduler
# ranks 2+ are dedicated to workers
from asyncevo import initialize
from asyncevo import AsyncGa
import numpy as np
from math import inf
from asyncevo import Member
from pathlib import Path


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
    """Rosenbrock-like test fitness function"""
    n = len(x)
    if n < 2:
        raise ValueError('dimension must be greater than one')
    return -sum(100 * (x[i]**2 - x[i+1])**2 + (x[i] - 1)**2
                for i in range(n-1))


def main():
    ga = AsyncGa(initial_state=np.random.randn(10),
                 population_size=20,
                 scheduler=initialize.mpi_scheduler,
                 global_seed=21048,
                 sigma=1.1,
                 cooling_factor=1.0,
                 annealing_start=0,
                 annealing_stop=inf,
                 table_size=20000,
                 max_table_step=2,
                 member_type=Member,
                 save_filename=Path("test.asyncga"))
    ga.run(sphere, 10)


if __name__ == "__main__":
    main()
