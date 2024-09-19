import numpy as np
from cpp_modules import column_cholesky_cpp
from cpp_torch_modules import cholesky_torch
print("succesful import")
from torch_modules.generate_data import generate_simple_random_psd_matrix

########## check if the C++ code works ##########
n = 1000
A = generate_simple_random_psd_matrix(n=n, diag_noise_factor=1.1).numpy()

# Convert to list of lists for our C++ function (modify in-place)
A_list = A.tolist()

# Perform Cholesky decomposition
L_list = column_cholesky_cpp.cholesky_decomposition(A_list)

# Convert back to numpy array
L = np.array(L_list)

# take only the lower triangular part
# this is necessary for the cholesky to work

L = np.tril(L)
if n < 11:
    print("L = ")
    print(L)
    print("\n")

    print("A = ")
    print(A)

# Verify the decomposition
print("\nVerification: L * L^T should equal the original matrix:")
print("max and min difference between L * L^T and A:")
print(np.max(np.dot(L, L.T) - A))
print(np.min(np.dot(L, L.T) - A))

assert np.allclose(np.dot(L, L.T), A, atol=1e-15)

print("\nCholesky decomposition successful!")