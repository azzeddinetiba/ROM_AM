import os
from setuptools import find_packages, setup


setup(
   name='ROM_AM',
   version='0.1',
   description='Non-intrusive Reduced Order Modeling packages',
   author='TIBA Azzeddine',
   author_email='azzeddine.tiba@lecnam.net',
   packages=['rom_am'], 
   install_requires=['numpy', 'scipy',], #external packages as dependencies
)
