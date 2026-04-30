import numpy as np
import matplotlib.pyplot as plt

from least_squares import least_squares
from regularized_ls import regularized_ls

# 1.  Generate synthetic data
np.random.seed(42)

# Class 1
X1 = np.random.randn(50, 2) + [2, 2]
y1 = np.ones(50)

# Class -1
X2 = np.random.randn(50, 2) + [-2, -2]
y2 = -np.ones(50)

# Combine data
X = np.vstack((X1, X2))
y = np.hstack((y1, y2))

# 2. train models
x_ls = least_squares(X, y)
x_reg = regularized_ls(X, y, lam=0.1)

# 3. Predictions
pred_ls = np.sign(X @ x_ls)
pred_reg = np.sign(X @ x_reg)

# 4. Accuracy
acc_ls = np.mean(pred_ls == y)
acc_reg = np.mean(pred_reg == y)

print("Accuracy LS:", acc_ls)
print("Accuracy Regularized:", acc_reg)

# 5. Graph
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
plt.title("classification with Least Squares and Regularized Least Squares")
plt.show()