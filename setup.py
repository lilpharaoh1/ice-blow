# setup.py
from setuptools import setup, find_packages

setup(
    name="ice_blow_env",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "numpy",
        "pygame",
    ],
)

