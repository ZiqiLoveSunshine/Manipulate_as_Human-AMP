from setuptools import find_packages
from distutils.core import setup

setup(
    name='arm_gym',
    version='1.0.0',
    author='Changda Tian',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='deepfluency@sjtu.edu.cn',
    description='Isaac Gym environments for Robot Arms',
    install_requires=['isaacgym',
                      'rsl-rl',
                      'matplotlib']
)