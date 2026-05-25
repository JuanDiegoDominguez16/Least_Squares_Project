"""
Least squares classifier for vehicle operational-failure risk.

Based on Boyd & Vandenberghe (2018), chapter 14, sections 14.1 and 14.2.
Two classifiers are fitted on the same data:

  1.  Plain least squares  (Ch. 14)
  2.  Ridge regularized LS (Ch. 15, sec. 15.3.1) for numerical
      stability and improved generalization.

Pipeline
--------
1. Load the CSV with the 100 vehicles.
2. Select the six numerical specification columns as features.
   (We deliberately exclude `Riesgo_Aceleracion`, `Riesgo_Consumo`,
    `Riesgo_Antiguedad` and `Condiciones_Riesgo_Cumplidas`: the label
    is derived from those columns, so using them would be data
    leakage.)
3. Standardize each feature column to zero mean and unit variance.
4. Add an intercept column of ones to obtain the design matrix A.
5. Solve A^T A x = A^T b           (ordinary LS).
6. Solve (A^T A + lam I) x = A^T b (Ridge).
7. Predict class labels as sign( A x ) and report accuracy,
   confusion matrix, and coefficient norm.
8. Plot the histogram of the raw classifier scores by true class,
   following the standard visualization in Boyd & Vandenberghe Ch. 14.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from least_squares import least_squares
from regularized_ls import regularized_ls


# ----------------------------------------------------------------------
# 1.  Load the dataset
# ----------------------------------------------------------------------
# The CSV is encoded in latin-1 because of the character 'ñ' in 'Año'.
CSV_PATH = Path("data/dataset_carros_clasificacion.csv")
df = pd.read_csv(CSV_PATH, encoding="latin-1")

FEATURE_COLS = [
    "Año",
    "Motor_cc",
    "Aceleracion_0_100_seg",
    "Consumo_L_100km",
    "Capacidad_Tanque_L",
    "Autonomia_km",
]
LABEL_COL = "Riesgo_Clasificacion"

X_raw = df[FEATURE_COLS].to_numpy(dtype=float)   # (m, 6)
y     = df[LABEL_COL].to_numpy(dtype=float)      # (m,) values in {-1, +1}
m, n_feat = X_raw.shape


# ----------------------------------------------------------------------
# 2.  Standardize each column to zero mean and unit variance.
#     Without this step, variables with large magnitudes (e.g. Año,
#     Motor_cc) would dominate the squared error and the fit would be
#     biased away from variables with smaller magnitudes (e.g.
#     Aceleracion, Consumo).  See Boyd & Vandenberghe (2018), sec. 13.3.
# ----------------------------------------------------------------------
mu    = X_raw.mean(axis=0)
sigma = X_raw.std(axis=0, ddof=0)
X_std = (X_raw - mu) / sigma


# ----------------------------------------------------------------------
# 3.  Add the intercept column.  The classifier is
#         f_tilde(x) = beta_0 + beta_1 x_1 + ... + beta_6 x_6 ,
#     and the final decision rule is  f(x) = sign( f_tilde(x) ).
# ----------------------------------------------------------------------
A = np.hstack([np.ones((m, 1)), X_std])          # (m, 7)


# ----------------------------------------------------------------------
# 4.  Fit both classifiers
# ----------------------------------------------------------------------
LAMBDA = 1.0
beta_ls  = least_squares(A, y)
beta_reg = regularized_ls(A, y, lam=LAMBDA)


# ----------------------------------------------------------------------
# 5.  Predictions and metrics
# ----------------------------------------------------------------------
score_ls  = A @ beta_ls
score_reg = A @ beta_reg
pred_ls   = np.sign(score_ls)
pred_reg  = np.sign(score_reg)


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred):
    """Return the 2x2 confusion matrix with rows = true class, cols = pred class,
    ordered as (+1, -1)."""
    tp = int(np.sum((y_true ==  1) & (y_pred ==  1)))
    fn = int(np.sum((y_true ==  1) & (y_pred == -1)))
    fp = int(np.sum((y_true == -1) & (y_pred ==  1)))
    tn = int(np.sum((y_true == -1) & (y_pred == -1)))
    return np.array([[tp, fn],
                     [fp, tn]])


acc_ls,  acc_reg  = accuracy(y, pred_ls),  accuracy(y, pred_reg)
cm_ls,   cm_reg   = confusion_matrix(y, pred_ls), confusion_matrix(y, pred_reg)
norm_ls, norm_reg = np.linalg.norm(beta_ls), np.linalg.norm(beta_reg)


# ----------------------------------------------------------------------
# 6.  Report
# ----------------------------------------------------------------------
def fmt_cm(cm):
    return ("                 pred +1   pred -1\n"
            f"   true +1   {cm[0,0]:>7d}   {cm[0,1]:>7d}\n"
            f"   true -1   {cm[1,0]:>7d}   {cm[1,1]:>7d}")


print(f"Dataset: {m} samples, {n_feat} features\n")

print("=" * 60)
print("Least squares classifier")
print("=" * 60)
print(f"Accuracy        : {acc_ls:.4f}")
print(f"||beta||_2      : {norm_ls:.4f}")
print(f"Confusion matrix:\n{fmt_cm(cm_ls)}")

print()
print("=" * 60)
print(f"Ridge classifier  (lambda = {LAMBDA})")
print("=" * 60)
print(f"Accuracy        : {acc_reg:.4f}")
print(f"||beta||_2      : {norm_reg:.4f}")
print(f"Confusion matrix:\n{fmt_cm(cm_reg)}")

print()
print("Coefficients in the standardized feature space")
print("-" * 60)
header = ["intercept"] + FEATURE_COLS
print(f"  {'variable':<25s}  {'LS':>10s}  {'Ridge':>10s}")
for name, b_ls, b_reg in zip(header, beta_ls, beta_reg):
    print(f"  {name:<25s}  {b_ls:+10.4f}  {b_reg:+10.4f}")


# ----------------------------------------------------------------------
# 7.  Visualization: histogram of raw classifier scores by true class
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, score, title in [
    (axes[0], score_ls,  "Least squares"),
    (axes[1], score_reg, f"Ridge ($\\lambda$={LAMBDA})"),
]:
    ax.hist(score[y ==  1], bins=18, alpha=0.65, label="True $+1$", color="#c0392b")
    ax.hist(score[y == -1], bins=18, alpha=0.65, label="True $-1$", color="#2980b9")
    ax.axvline(0, color="k", linestyle="--", linewidth=1, label="Decision boundary")
    ax.set_xlabel(r"Score $\tilde f(x) = a^T \beta$")
    ax.set_title(title)
    ax.legend(loc="upper left")
axes[0].set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("classifier_scores.png", dpi=150, bbox_inches="tight")
plt.show()