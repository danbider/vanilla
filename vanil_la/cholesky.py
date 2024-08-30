def cholesky(A):
    """
    Cholesky decomposition of a symmetric positive definite matrix A.
    naive implementation.
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
