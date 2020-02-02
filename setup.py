from setuptools import setup

setup(name='asyncevo',
      version='0.0.1',
      description='Library of asyncronous distributed evolutionary algorithms',
      author='Nathaniel J. Rodriguez',
      packages=['asyncevo'],
      url='https://github.com/Nathaniel-Rodriguez/asyncevo.git',
      install_requires=[
          'dask>=2.10.1',
      ],
      include_package_data=True)
