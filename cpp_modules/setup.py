from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'column_cholesky_cpp',
        ['cholesky_module.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++'
    ),
]

setup(
    name='cholesky_module',
    version='0.0.1',
    author='Your Name',
    author_email='your.email@example.com',
    description='A Cholesky decomposition module',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    install_requires=['pybind11>=2.5.0'],
)