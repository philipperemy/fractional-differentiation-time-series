from setuptools import setup, find_packages

VERSION = '1.1'

setup(
    name='fracdiff2',
    version=VERSION,
    description='Fractional Differentiation framework for time series.',
    author='Philippe Remy',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib'
    ]
)
