import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# -----------------------------
# Load Dataset
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "insurance.csv"

insurance_data = pd.read_csv(DATA_PATH)

X = insurance_data["age"]
Y = insurance_data["charges"]

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

X_train = X_train.to_numpy(dtype=np.float64)
X_test = X_test.to_numpy(dtype=np.float64)

Y_train = Y_train.to_numpy(dtype=np.float64)
Y_test = Y_test.to_numpy(dtype=np.float64)

# -----------------------------
# Hyperparameters
# -----------------------------
lr = 0.0005
iterations = 1000000
n = len(X_train)

# -----------------------------
# Load Previous Weights
# -----------------------------
if os.path.exists("weights.npy") and os.path.exists("bias.npy"):
    m = np.load("weights.npy")[0]
    c = np.load("bias.npy")[0]
    print("Loaded previous model.")
else:
    m = 1.0
    c = 0.0
    print("Starting from scratch.")

# -----------------------------
# Store Cost History
# -----------------------------
cost_history = []

# -----------------------------
# Gradient Descent
# -----------------------------
for i in range(iterations):

    y_pred = m * X_train + c

    residual = y_pred - Y_train

    cost = np.sum(residual ** 2) / (2 * n)

    cost_history.append(cost)

    dm = np.sum(residual * X_train) / n
    dc = np.sum(residual) / n

    m -= lr * dm
    c -= lr * dc

    if i % 1000 == 0:
        print(f"Iteration {i:7d} | Cost = {cost:.4f}")

# -----------------------------
# Save Model
# -----------------------------
np.save("weights.npy", np.array([m]))
np.save("bias.npy", np.array([c]))

print("\nModel Saved Successfully!")

print("\nSlope (m):")
print(m)

print("\nIntercept (c):")
print(c)

# ===========================================================
# VISUALIZATIONS
# ===========================================================

os.makedirs("images", exist_ok=True)

# -----------------------------
# 1. Cost Curve
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(cost_history)

plt.title("Cost vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.grid(True)

plt.tight_layout()
plt.savefig("images/cost_curve.png")
plt.close()

# -----------------------------
# 2. Dataset + Regression Line
# -----------------------------
plt.figure(figsize=(8, 5))

plt.scatter(
    X_test,
    Y_test,
    alpha=0.7,
    label="Actual Data"
)

x_line = np.linspace(X_test.min(), X_test.max(), 100)
y_line = m * x_line + c

plt.plot(
    x_line,
    y_line,
    linewidth=2,
    color="red",
    label="Regression Line"
)

plt.title("Linear Regression Fit")
plt.xlabel("Age")
plt.ylabel("Insurance Charges")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("images/regression_line.png")
plt.close()

print("\nGraphs saved successfully!")
print("images/cost_curve.png")
print("images/regression_line.png")