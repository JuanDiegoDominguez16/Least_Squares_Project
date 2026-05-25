# Least Squares Project

A machine learning and applied linear algebra project focused on binary classification using **Ordinary Least Squares (OLS)** and **Ridge Regularization**.

The project predicts whether a vehicle presents a **high operational failure risk** based on technical and performance specifications such as fuel consumption, acceleration, engine displacement, autonomy, and manufacturing year.

This implementation was developed as part of an applied linear algebra project inspired by the methodology presented in:

* *Boyd & Vandenberghe — Introduction to Applied Linear Algebra*

---

# Project Overview

The main goal of this project is to:

* Formulate binary classification as a least squares problem
* Solve the normal equations numerically
* Interpret the solution geometrically as an orthogonal projection
* Compare standard least squares against Ridge regularization
* Analyze coefficient behavior and classification performance

The classifier assigns each vehicle to one of two classes:

* `1` → High operational risk
* `-1` → Low operational risk

---

# Mathematical Formulation

Given a design matrix:
$
[
X \in \mathbb{R}^{m \times n}
]
$

and a label vector:

[
y \in {-1,1}^m
]

we seek a parameter vector:

[
\beta
]

that minimizes the squared error:

[
\min_{\beta} ||X\beta - y||^2
]

The solution is obtained through the normal equations:

[
X^TX\beta = X^Ty
]

and solved numerically using:

```python
numpy.linalg.solve
```

instead of explicitly computing the matrix inverse.

---

# Ridge Regularization

To improve numerical stability and reduce overfitting, the project also implements Ridge regression:

[
\min_{\beta} ||X\beta - y||^2 + \lambda ||\beta||^2
]

which leads to the regularized system:

[
(X^TX + \lambda I)\beta = X^Ty
]

This reduces coefficient magnitude and improves generalization.

---

# Dataset

The dataset contains 100 vehicles described by technical specifications.

## Features Used

| Variable                | Description                  |
| ----------------------- | ---------------------------- |
| `Año`                   | Manufacturing year           |
| `Motor_cc`              | Engine displacement          |
| `Aceleracion_0_100_seg` | 0–100 km/h acceleration time |
| `Consumo_L_100km`       | Fuel consumption             |
| `Capacidad_Tanque_L`    | Fuel tank capacity           |
| `Autonomia_km`          | Estimated autonomy           |

## Target Variable

| Variable               | Meaning           |
| ---------------------- | ----------------- |
| `Riesgo_Clasificacion` | Binary risk label |

---

# Project Structure

```text
Least_Squares_Project/
│
├── data/
│   └── vehicles.csv
│
├── least_squares.py
├── regularized_ls.py
├── main.py
├── requirements.txt
└── README.md
```

---

# Implementation Details

The pipeline consists of:

1. Loading the dataset with `pandas`
2. Selecting predictor variables
3. Standardizing the features
4. Adding an intercept column
5. Solving the normal equations
6. Predicting vehicle risk
7. Evaluating the classifier

---

# Feature Standardization

All variables are standardized before training:

[
z = \frac{x-\mu}{\sigma}
]

This prevents variables with larger magnitudes from dominating the optimization process.

---

# Geometric Interpretation

One of the key theoretical ideas of the project is the geometric interpretation of least squares.

The prediction vector:

[
X\hat{\beta}
]

is the orthogonal projection of the label vector:

[
y
]

onto the column space of the design matrix:

[
Col(X)
]

This interpretation connects machine learning classification with fundamental concepts from linear algebra.

---

# Evaluation Metrics

The project evaluates both classifiers using:

* Accuracy
* Confusion matrix
* Euclidean norm of the coefficient vector

---

# Results Summary

| Model                  | Accuracy |
| ---------------------- | -------- |
| Ordinary Least Squares | 93%      |
| Ridge Regression       | 95%      |

Ridge regularization produced:

* Smaller coefficient norms
* Better numerical stability
* Slightly better classification performance

---

# Installation

Clone the repository:

```bash
git clone https://github.com/JuanDiegoDominguez16/Least_Squares_Project.git
```

Move into the project directory:

```bash
cd Least_Squares_Project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Run the main file:

```bash
python main.py
```

---

# Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

---

# Key Concepts

This project integrates concepts from:

* Linear algebra
* Least squares optimization
* Orthogonal projections
* Binary classification
* Ridge regularization
* Numerical linear algebra
* Machine learning fundamentals

---

# Future Improvements

Possible extensions include:

* Train/test split
* Cross-validation
* Multiclass classification
* Logistic regression comparison
* Robust regression methods
* Visualization of the decision boundary

---

# References

* Boyd, S., & Vandenberghe, L. *Introduction to Applied Linear Algebra: Vectors, Matrices, and Least Squares*. Cambridge University Press.
* Tikhonov, A. N., & Arsenin, V. Y. *Solutions of Ill-Posed Problems*.
* Mobley, R. K. *An Introduction to Predictive Maintenance*.

---

# Author

Juan Diego Dominguez

