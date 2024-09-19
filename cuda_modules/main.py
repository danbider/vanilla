import torch
import os
import numpy as np
import cholesky_cuda
import time


os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # for debugging

def generate_well_conditioned_spd(n, condition_number=10, random_state=None):
    """
    Generate a well-conditioned, symmetric positive definite matrix.

    Parameters:
    n (int): Size of the square matrix
    condition_number (float): Desired condition number (default: 10)
    random_state (int or None): Seed for random number generator

    Returns:
    numpy.ndarray: A well-conditioned SPD matrix
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate random orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(n, n))

    # Generate eigenvalues
    lambda_min = 1
    lambda_max = condition_number
    lambdas = np.exp(np.linspace(np.log(lambda_min), np.log(lambda_max), n))

    # Create diagonal matrix of eigenvalues
    D = np.diag(lambdas)

    # Create SPD matrix
    A = Q @ D @ Q.T

    return A

# Generate a well-conditioned, symmetric positive definite matrix
n = 1000
A = torch.tensor(generate_well_conditioned_spd(n, condition_number=10, random_state=0), dtype=torch.float32)

print("performing Cholesky decomposition on GPU")

# time the Cholesky decomposition
start = time.time()
# Perform Cholesky decomposition
L = cholesky_cuda.cholesky_decomposition(A.cuda())

torch.cuda.synchronize()
end = time.time()
print(f"Time taken: {end - start:.4f} seconds")

L = L.cpu()

# # save L in a csv file no header
# np.savetxt("L.csv", L.numpy(), delimiter=",")
# LAST_X = 10
# print(f"LAST {LAST_X}X{LAST_X} of L:")
# print(L[-LAST_X:, -LAST_X:])
# print("success")

# print(f"LAST {LAST_X}X{LAST_X} of outer product:")
# print((L@L.T)[-LAST_X:, -LAST_X:])
# outer = L@L.T
# FIRST_X = 3

# print(torch.max(outer[:FIRST_X, :FIRST_X] - A[:FIRST_X, :FIRST_X]))
# print(torch.min(outer[:FIRST_X, :FIRST_X] - A[:FIRST_X, :FIRST_X]))

assert torch.allclose(L@L.T, A, atol=1e-6)

print("test passed")

# time the Cholesky decomposition using torch.linalg.cholesky
print("performing Cholesky decomposition on GPU")
start = time.time()
L_torch = torch.linalg.cholesky(A)
torch.cuda.synchronize()
end = time.time()
print(f"Time taken for torch.linalg: {end - start:.4f} seconds")

# time the Cholesky decomposition
start = time.time()
# Perform Cholesky decomposition
L = cholesky_cuda.cholesky_decomposition(A.cuda())

torch.cuda.synchronize()
end = time.time()
print(f"Time taken for my implementation: {end - start:.4f} seconds")

# torch linalg on cpu
start = time.time()
L_torch = torch.linalg.cholesky(A.cuda())
end = time.time()
print(f"Time taken for torch.linalg on CUDA: {end - start:.4f} seconds")



