from setuptools import setup


setup(name='asyncevo',
      version='0.0.1',
      description='Library of asyncronous distributed evolutionary algorithms',
      author='Nathaniel J. Rodriguez',
      packages=['asyncevo'],
      url='https://github.com/Nathaniel-Rodriguez/asyncevo.git',
      install_requires=[
          'dask>=2.10.1<3',  # maybe more than 3 and less than 2.10, probs 2
          'dask-core>=2.10.1<3'
          'distributed>=2.10.0',
          'numpy>=1.18.1',  # likely compatible with earlier versions.
          'dask-mpi>=2.0.0'
          # sphinx?
      ],
      python_requires='>=3.7',
      include_package_data=True)
