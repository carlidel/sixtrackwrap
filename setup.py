import os
import re
import sys
import platform
import subprocess

from setuptools import setup

__version__ = '0.0.1'

setup(
    name='sixtrackwrap',
    version=__version__,
    author='Carlo Emilio Montanari',
    author_email='carlidel95@gmail.com',
    description='basic sixtracklib for my personal needs',
    packages=["sixtrackwrap"],
    install_requires=['numba', 'numpy'],
    setup_requires=['numba', 'numpy'],
    license='MIT',
    include_package_data=True
)
