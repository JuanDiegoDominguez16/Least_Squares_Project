import numpy as np

# Solves the least squares problem Ax = b for x. Is the basic implementation for the project, this will train the ai.

def least_squares(A, b):
    return np.linalg.pinv(A) @ b

# Example usage:
A = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([1, 2, 3])

x = least_squares(A, b)

print("x:", x)
print("Ax:", A @ x)
print("b:", b)
# Output: [0. 0.5]