# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

#with open('LICENSE') as f:
#    license = f.read()

setup(
    name='graph-dynamics',
    version='1.0.0',
    description='Studying graph dynamics',
    long_description=readme,
    author='Kostadin Cvejoski',
    author_email='cvejoski@gmail.com',
    #url='https://github.com/kennethreitz/samplemod',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=['numpy', 'matplotlib', 'scipy', 'networkx', 'pandas']
)
