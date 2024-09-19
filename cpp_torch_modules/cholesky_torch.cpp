#include <torch/extension.h>
#include <vector>
#include <cmath>

// Cholesky calling the torch cpp extension functions
torch::Tensor torch_cholesky_naive_variant_3(torch::Tensor matrix) {
    // Check if the matrix is empty and exit if it is
    if (matrix.numel() == 0) {
        return matrix;
    }

    int rows = matrix.size(0);
    int cols = matrix.size(1);

    // if matrix is 1x1 (base case) take square root of the 0,0 element
    if (rows == 1 && cols == 1) {
        matrix[0][0] = torch::sqrt(matrix[0][0]);
        return matrix;
    }

    // Create a copy of the input tensor to modify
    auto result = matrix.clone();

    // Loop through each column
    for (int col = 0; col < cols; ++col) {
        // l_{11}: Take the square root of the diagonal element
        result[col][col] = torch::sqrt(result[col][col]);

        // l_{21}: Loop through each row below the diagonal within the column
        // and divide each element by the diagonal element
        for (int row = col + 1; row < rows; ++row) {
            result[row][col] /= result[col][col];
        }

        // l_{22} = A_{22} - l_{21} * l_{21}^T
        // a double for loop to update the bottom right submatrix
        for (int row22 = col + 1; row22 < rows; ++row22) {
            for (int col22 = col + 1; col22 < cols; ++col22) {
                // (uu^T)_{ij} = u_i * u_j 
                result[row22][col22] -= result[row22][col] * result[col22][col];
            }
        }
    } // end of main loop through each column
    
    // take the lower triangular part of the result
    result = torch::tril(result);

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cholesky_decomposition", &torch_cholesky_naive_variant_3, "A function which performs Cholesky decomposition using PyTorch");
}