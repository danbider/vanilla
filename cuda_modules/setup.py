from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cholesky_cuda',
    ext_modules=[
        CUDAExtension('cholesky_cuda', [
            'cholesky_cuda.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })