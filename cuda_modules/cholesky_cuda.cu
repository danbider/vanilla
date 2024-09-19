#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

#define CHECK_CUDA(x) do { \
  cudaError_t status = (x); \
  if (status != cudaSuccess) { \
    std::printf("CUDA error: %s at line %d\n", cudaGetErrorString(status), __LINE__); \
    std::exit(EXIT_FAILURE); \
  } \
} while(0)

__global__ void column_kernel(float *matrix, int rows, int cols, int col_id) {
    __shared__ float element0;
    if(threadIdx.x == 0) {
        element0 = sqrt(matrix[col_id * cols + col_id]);
        matrix[col_id * cols + col_id] = element0;
    }
    __syncthreads(); // all threads must wait for thread 0 to finish writing shared variable element0
    for(int r = col_id+1 + threadIdx.x; r < rows; r += blockDim.x) {
        matrix[r*cols + col_id] = matrix[r*cols + col_id] / element0;
    }
}
__global__ void update_submatrix_kernel(float *matrix, int rows, int cols, int col_id) {
    for(int r = col_id+1 + blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += blockDim.x * gridDim.x) {
        for(int c = col_id+1 + blockIdx.y * blockDim.y + threadIdx.y; c < cols; c += blockDim.y * gridDim.y) {
            matrix[r*cols + c] -= matrix[r*cols + col_id] * matrix[c*cols + col_id];
        }
    }
}

torch::Tensor torch_cholesky_naive_variant_3_cuda(torch::Tensor matrix) {
    auto result = matrix.to(torch::kCUDA, torch::kFloat32).clone();
    
    int rows = result.size(0);
    int cols = result.size(1);
    float* matrix_ptr = result.data_ptr<float>();

    const int max_threads_per_block = 1024;

    for(int col = 0; col < cols; col++) {
        column_kernel<<<1, 1024>>>(matrix_ptr, rows, cols, col);
        dim3 grid2(16, 16);
        dim3 block2(32, 32);
        update_submatrix_kernel<<<grid2, block2>>>(matrix_ptr, rows, cols, col);
    }
    cudaDeviceSynchronize();

    return torch::tril(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cholesky_decomposition", &torch_cholesky_naive_variant_3_cuda, "A function which performs Cholesky decomposition using CUDA");
}