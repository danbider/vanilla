import torch
import numpy as np


def cholesky(A):
    """
    Cholesky decomposition of a symmetric positive definite matrix A.
    naive implementation in numpy
    Args:
        A (np.ndarray): Symmetric positive definite matrix
    Returns:
        np.ndarray: Lower triangular matrix L such that A = LL^T
    """
    n = A.shape[0]
    L = np.zeros(A.shape)
    for j in range(n):  # loop over columns
        v_j_to_n = np.copy(A[j:, j])  # vector from j-th element to the end, size n-j
        for k in range(j):
            v_j_to_n -= L[j, k] * L[j:, k]  # n-j vectors
        L[j:n, j] = v_j_to_n / np.sqrt(v_j_to_n[0])
    return L


# def cholesky(A):  # slightly different suggestion from claude
#     n = A.shape[0]
#     L = np.zeros(A.shape)
#     for j in range(n):  # loop over columns
#         v_j_to_n = A[j:, j].copy()  # vector from j-th element to the end, size n-j
#         for k in range(j):
#             v_j_to_n -= L[j:, k] * L[j, k]
#         L[j, j] = np.sqrt(v_j_to_n[0])  # Diagonal element
#         L[j + 1 :, j] = v_j_to_n[1:] / L[j, j]  # Off-diagonal elements
#     return L


def cholesky_variant_three(A):
    """Cholesky decomposition of a positive definite matrix, variant 3, column by column.
    with in-place operations.

    Args:
        A (torch.Tensor): symmetric positive definite matrix, shape (n, n)

    Returns:
        torch.Tensor: the Cholesky factor, i.e., lower triangular matrix L such that A = LL^T
    """
    A = torch.tril(
        A
    )  # ignore the upper triangular part which is identical to the lower triangular part
    n = A.shape[0]  # dimension of the matrix
    if n == 1:
        # base case, A is 1x1 matrix
        A[0, 0] = torch.sqrt(A[0, 0])
        return A

    for j in range(n):
        # overwrite the top diagonal element (scalar block 1,1), take the square root
        A[j, j] = torch.sqrt(A[j, j])
        # overwrite the column below the diagonal element (vector block 2,1), divide by the diagonal element
        A[j + 1 :, j] /= A[j, j]
        # update the bottom-right submatrix (matrix block 2,2), subtract the outer product of the column computed above
        # the cholesky factorization of the bottom-right submatrix is computed recursively, will be done in the next iteration.
        A[j + 1 :, j + 1 :] -= torch.tril(torch.outer(A[j + 1 :, j], A[j + 1 :, j]))
        # i.e., next iteration will be $Cholesky(A[j + 1 :, j + 1 :])$ of this new submatrix.
    return A


def choelesky_variant_three_blocked(A: torch.Tensor, block_size: int):
    """Blocked variant 3 of the Cholesky decomposition of a positive definite matrix.

    Args:
        A (torch.Tensor): symmetric positive definite matrix, shape (n, n)
        block_size (int): block size

    Returns:
        (torch.Tensor): A modified in-place to contain the Cholesky factor L
    """
    n = A.shape[0]
    # start by working just on the lower triangular part
    A = torch.tril(A)
    # loop and partition the matrix into blocks
    for j in range(0, n, block_size):
        # print(f"partition number j={j}")

        # L_11 is the cholesky of the original block (1, 1)
        # we need it for the next two blocks
        # start by the non-blocked cholesky variant 3, swap to blocked later
        # (block_size, block_size) block
        L_11 = cholesky_variant_three(A[j : j + block_size, j : j + block_size])
        A[j : j + block_size, j : j + block_size] = L_11

        # L_{21} involves solving the lower triangular system with multiple columns
        # (block_size, n-j-block_size) block
        L_21 = torch.linalg.solve_triangular(
            L_11, A[j + block_size : n, j : j + block_size].T, upper=False
        ).T
        # modify A in place
        A[j + block_size : n, j : j + block_size] = L_21

        # L_{22} involves the cholesky (of the Schur complement? check)
        L_22 = torch.tril(A[j + block_size : n, j + block_size : n] - L_21 @ L_21.T)
        A[j + block_size : n, j + block_size : n] = L_22

    return A


def choelesky_variant_three_blocked_lazy(A: torch.Tensor, block_size: int):
    """Blocked variant 3 of the Cholesky decomposition of a positive definite matrix.

    Args:
        A (torch.Tensor): symmetric positive definite matrix, shape (n, n)
        block_size (int): block size

    Returns:
        (torch.Tensor): A modified in-place to contain the Cholesky factor L
    """
    L_21_list = []
    n = A.shape[0]
    # start by working just on the lower triangular part
    A = torch.tril(A)
    # loop and partition the matrix into blocks
    for j in range(0, n, block_size):
        print(f"partition number j={j}")

        # L_11 is the cholesky of the original block (1, 1)
        # we need it for the next two blocks
        # start by the non-blocked cholesky variant 3, swap to blocked later
        # (block_size, block_size) block
        L_11 = cholesky_variant_three(A[j : j + block_size, j : j + block_size])
        A[j : j + block_size, j : j + block_size] = L_11

        # L_{21} involves solving the lower triangular system with multiple columns
        # (block_size, n-j-block_size) block
        L_21 = torch.linalg.solve_triangular(
            L_11, A[j + block_size : n, j : j + block_size].T, upper=False
        ).T
        # modify A in place
        A[j + block_size : n, j : j + block_size] = L_21

        # L_{22} involves the cholesky (of the Schur complement? check). the below worked
        # L_22 = torch.tril(A[j + block_size : n, j + block_size : n] - L_21 @ L_21.T)
        # A[j + block_size : n, j + block_size : n] = L_22

        # experimental below:
        # instead of the above, we modify just A_{22}[:, one block size ahead] and keep trak of L_21 to be used in the next iteration
        # (n-j-block_size, block_size) block
        A[j + block_size : n, j + block_size : j + 2 * block_size] -= (
            L_21 @ L_21.T[:, :block_size]
        )  # current block
        # now subtract all the previous L_21s at the respective positions.
        # the positions depend on the block size and the index of the L_21 in the list
        for i, L_21_dict in enumerate(L_21_list):
            old_block_counter = L_21_dict["block_counter"]
            old_L_21 = L_21_dict["matrix"]
            # TODO: might be one index back, check
            # A[j + block_size : n, j + block_size : j + 2 * block_size] -= (
            result = (
                old_L_21
                @ old_L_21.T[
                    :, j - old_block_counter : j - old_block_counter + block_size
                ]
            )
            # take the relevant rows of the result to update the next A_{22} block
            A[j + block_size : n, j + block_size : j + 2 * block_size] -= result[
                j - old_block_counter :, :
            ]

        L_21_list.append({"matrix": L_21, "block_counter": j})

    return A
