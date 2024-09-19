import torch
import os
import numpy as np
import cholesky_torch 


print("success")

n = 10
# create a random symmetric positive-definite matrix
A = np.random.rand(n, n)
A = np.dot(A, A.T)
# Add a small diagonal to ensure positive-definiteness
A += np.eye(n) * 0.1
A = torch.tensor(A).float()

# Perform Cholesky decomposition
L = cholesky_torch.cholesky_decomposition(A)

# Verify the decomposition
print("\nVerification: L * L^T should equal the original matrix:")
print("max and min difference between L * L^T and A:")

print(torch.max(L@L.T - A))
print(torch.min(L@L.T - A))

assert torch.allclose(L@L.T, A, atol=1e-15)
