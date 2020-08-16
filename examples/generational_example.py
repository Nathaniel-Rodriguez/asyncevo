from asyncevo import LocalScheduler
from asyncevo import GenerationalGa
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
    with LocalScheduler({'n_workers': 10,
                         'threads_per_worker': 1}) as local_scheduler:
        ga = GenerationalGa(
            initial_state=np.array([1.4, 0.3, -0.25, 1.01]),
            population_size=10,
            scheduler=local_scheduler,
            global_seed=96879,
            sigma=0.01,
            cooling_factor=1.0,
            annealing_start=0,
            annealing_stop=inf,
            table_size=20000,
            max_table_step=1,
            member_type=Member,
            save_filename=Path("test.ga"),
            save_every=1000)
        ga.run(sphere, 30, take_member=False)

        # load pop from file and continue
        ga = GenerationalGa.from_file("test.ga",
                                      scheduler=local_scheduler,
                                      global_seed=432,
                                      sigma=0.01,
                                      cooling_factor=1.0,
                                      save_filename="test.ga")
        ga.run(sphere, 10, take_member=False)


if __name__ == "__main__":
    main()
