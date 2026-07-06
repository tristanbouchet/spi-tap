from setuptools import find_packages, setup

setup(
    name = 'spitap',
    packages = find_packages(include = ['spitap']),
    version = '0.1',
    description = 'Tools to analyze INTEGRAL/SPI transients data',
    author = 'Tristan Bouchet',
    install_requires = ['numpy', 'pandas', 'scipy','astropy']
)