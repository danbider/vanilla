from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='cholesky_torch',
      ext_modules=[cpp_extension.CppExtension('cholesky_torch', ['cholesky_torch.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})