import numpy as np


def generate_structured_cg_inputs(n=100, problem_type="poisson"):
    """
    Generate structured inputs for conjugate gradient solver.

    Args:
    n (int): Size of the problem (n x n grid)
    problem_type (str): Type of problem to generate ('poisson' or 'toeplitz')

    Returns:
    A (np.array): Coefficient matrix
    b (np.array): Right-hand side vector
    x_true (np.array): True solution
    """

    if problem_type == "poisson":
        # 2D Poisson equation discretized on a square grid
        h = 1.0 / (n + 1)
        A = np.zeros((n**2, n**2))

        for i in range(n**2):
            A[i, i] = 4
            if i % n != 0:  # not leftmost column
                A[i, i - 1] = -1
            if (i + 1) % n != 0:  # not rightmost column
                A[i, i + 1] = -1
            if i >= n:  # not top row
                A[i, i - n] = -1
            if i < n * (n - 1):  # not bottom row
                A[i, i + n] = -1

        A /= h**2

        # Create a smooth solution
        x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        x_true = np.sin(np.pi * x) * np.sin(np.pi * y)
        x_true = x_true.flatten()

        # Compute right-hand side
        b = A @ x_true

    elif problem_type == "toeplitz":
        # Symmetric Toeplitz matrix
        column = np.zeros(n)
        column[0] = 2
        column[1] = -1
        A = np.zeros((n, n))
        for i in range(n):
            A[i, i:] = column[: n - i]
            A[i:, i] = column[: n - i]

        # Create a solution with alternating 1 and -1
        x_true = np.array([1 if i % 2 == 0 else -1 for i in range(n)])

        # Compute right-hand side
        b = A @ x_true

    else:
        raise ValueError("Invalid problem_type. Choose 'poisson' or 'toeplitz'.")

    return A, b, x_true


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


# # Example usage:
# n = 100
# cond = 10
# A = generate_well_conditioned_spd(n, condition_number=cond)
# print(f"Condition number: {np.linalg.cond(A)}")
# print(f"Is symmetric: {np.allclose(A, A.T)}")
# print(f"Is positive definite: {np.all(np.linalg.eigvals(A) > 0)}")
# plt.imshow(A)
