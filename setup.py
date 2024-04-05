from setuptools import setup, find_packages
dependencies = [module for module in open("requirements.txt").read().split("\n")]
setup(
    name='HC_feature_selection',
    version='0.1',
    packages=find_packages(exclude=['experiments']),
    install_requires=dependencies,
)
