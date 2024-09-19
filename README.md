# VANILLA: Custom kernels for faster numerical linear algebra

This repo implements PyTorch, C++, and CUDA kernels for the Cholesky decomposition, CG, and more soon. 

Currently, each kernel is in a separate folder:
* `cpp_modules/` for a plain C++ implementation
    * `cholesky_module.cpp`
* `cpp_torch_modules/` for a C++ implementation using the Pytorch C++ API working on tensors
    * `cholesky_torch.cpp`
* `cuda_modules/`
    * `cholesky_cuda.cu`

To run the kernels, `cd` to the folder and compile the module as follows. e.g., for running a `cuda` kernel, do
```
cd cuda_modules
python setup.py build_ext --inplace  
```

Then within the folder, run
```
python main.py
```
Which should call the kernel on some test data in python.