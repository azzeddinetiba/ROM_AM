from setuptools import find_packages, setup


setup(
    name='ROM_AM',
    version='0.1',
    description='Non-intrusive Reduced Order Modeling packages',
    author='TIBA Azzeddine',
    author_email='azzeddine.tiba@lecnam.net',
    packages=['rom_am'],
    # external packages as dependencies
    install_requires=['numpy >= 1.20.3', 'scipy >= 1.7.1',
                      'jax >= 0.2.26 ; platform_system=="linux"',
                      'jaxlib >= 0.1.75 ; platform_system=="linux"'],
)
