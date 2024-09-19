#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>

// naive implementation of Cholesky decomposition
// currently returning the modified matrix and not just changing in place. for learning purposes
std::vector<std::vector<double>> cholesky_naive_variant_3(std::vector<std::vector<double>> matrix) {
    // Check if the matrix is empty and exit if it is
    if (matrix.empty() || matrix[0].empty()) {
        return matrix;
    }

    int rows = matrix.size();
    int cols = matrix[0].size();

    // if matrix is 1x1 (base case) take square root of the 0,0 element
    if (rows == 1 && cols == 1) {
        matrix[0][0] = sqrt(matrix[0][0]);
        return matrix;
    }

    // Loop through each column
    for (int col = 0; col < cols; ++col) {
        // l_{11}: Take the square root of the diagonal element
        matrix[col][col] = sqrt(matrix[col][col]);

        // l_{21}: Loop through each row below the diagonal withing the column
        // and divide each element by the diagonal element
        for (int row = col + 1; row < rows; ++ row) {
            matrix[row][col] /= matrix[col][col];
        }

        // l_{22} = A_{22} - l_{21} * l_{21}^T
        // a double for loop to update the bottom right submatrix
        for (int row22 = col + 1; row22 < rows; ++row22) {
            for (int col22 = col + 1; col22 < cols; ++col22) {
                // (uu^T)_{ij} = u_i * u_j 
                matrix[row22][col22] -= matrix[row22][col] * matrix[col22][col];
            }
        }

    } // end of main loop through each column
    return matrix;
}


namespace py = pybind11;

PYBIND11_MODULE(column_cholesky_cpp, m) {
    m.doc() = "pybind11 Cholesky decomposition module";

    m.def("cholesky_decomposition", &cholesky_naive_variant_3, "A function which performs Cholesky decomposition");
}