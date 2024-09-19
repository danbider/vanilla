from setuptools import setup, find_packages, Extension
import pybind11
from torch.utils import cpp_extension


ext_modules = [
    Extension(
        'column_cholesky_cpp',
        ['cpp_modules/cholesky_module.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++'
    ),
    cpp_extension.CppExtension('cholesky_torch', ['cpp_modules/cholesky_torch.cpp'],        
        include_dirs=[cpp_extension.include_paths()],
        libraries=['c10'])]



setup(
    name="vanilla",
    version="0.1",
    packages=find_packages(),
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    install_requires=[
        "pybind11>=2.5.0", # pybind11 is a dependency of the C++ extension
        "torch",
        "numpy",
        "matplotlib"],
    cmdclass={'build_ext': cpp_extension.BuildExtension})
