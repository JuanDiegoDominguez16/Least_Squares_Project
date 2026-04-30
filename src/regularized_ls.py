import numpy as np

def regularized_ls(A, b, lam):
    n = A.shape[1] #number of variables
    I = np.eye(n) #Identity matrix of size n
    return np.linalg.solve(A.T @ A + lam * I, A.T @ b) #np.linalg.solve(A.T @ A + lam * I, A.T @ b) is more numerically stable than np.linalg.inv(A.T @ A + lam * I) @ A.T @ b, lam is the regularization parameter. This guarantees that the solution is more stable, the matrix is ALWAYS invertible when lam > 0 and less sensitive to noise in the data.

A = np.array([[1,2],[3,4],[5,6]])
b = np.array([1,2,3])

x = regularized_ls(A, b, lam=0.1)
print("x:", x)