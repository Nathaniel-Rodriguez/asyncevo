from setuptools import setup


setup(name='asyncevo',
      version='0.0.3',
      description='Library of asyncronous distributed evolutionary algorithms',
      author='Nathaniel J. Rodriguez',
      packages=['asyncevo'],
      url='https://github.com/Nathaniel-Rodriguez/asyncevo.git',
      install_requires=[
          'dask>=2.2<3',
          'distributed>=2.0<3',
          'numpy>=1.10.0',
          'dask-mpi>=2.0.0<3'
      ],
      python_requires='>=3.5',
      include_package_data=True)
