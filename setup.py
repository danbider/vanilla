from setuptools import setup, find_packages

setup(
    name="vanil-la",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
    ],
)
