import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# -----------------------------
# Load Dataset
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "insurance.csv"

insurance_data = pd.read_csv(DATA_PATH)

X = insurance_data[["age", "sex", "bmi", "children", "smoker", "region"]]
Y = insurance_data["charges"]

# -----------------------------
# One-Hot Encoding
# -----------------------------
X = pd.get_dummies(
    X,
    columns=["sex", "smoker", "region"],
    drop_first=True
)

X = X.astype(float)

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
lr = 0.0001
iterations = 10000000
n = len(X_train)

# -----------------------------
# Continue Training?
# -----------------------------
CONTINUE_TRAINING = True

# -----------------------------
# Load Previous Model
# -----------------------------
if (
    CONTINUE_TRAINING
    and os.path.exists("weights.npy")
    and os.path.exists("bias.npy")
    and os.path.exists("cost_history.npy")
):
    weights = np.load("weights.npy")
    bias = np.load("bias.npy")[0]
    cost_history = list(np.load("cost_history.npy"))

    print("Loaded previous model.")
    print(f"Previous Iterations: {len(cost_history):,}")

else:
    weights = np.ones(X_train.shape[1], dtype=np.float64)
    bias = 1.0
    cost_history = []

    print("Starting from scratch.")

# -----------------------------
# Gradient Descent
# -----------------------------
for i in range(iterations):

    Y_pred = X_train @ weights + bias

    residual = Y_pred - Y_train

    cost = np.sum(residual ** 2) / (2 * n)

    cost_history.append(cost)

    dw = (X_train.T @ residual) / n
    db = np.sum(residual) / n

    weights -= lr * dw
    bias -= lr * db

    if i % 10000 == 0:
        print(f"Iteration {i:7d} | Cost = {cost:.4f}")

# -----------------------------
# Save Model
# -----------------------------
# -----------------------------
# Save Model
# -----------------------------
np.save("weights.npy", weights)
np.save("bias.npy", np.array([bias]))
np.save("cost_history.npy", np.array(cost_history))
print("\nModel Saved Successfully!")

print("\nWeights:")
print(weights)

print("\nBias:")
print(bias)

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
# 2. Predicted vs Actual
# -----------------------------
Y_test_pred = X_test @ weights + bias

plt.figure(figsize=(8, 6))

plt.scatter(
    Y_test,
    Y_test_pred,
    alpha=0.7,
    label="Predictions"
)

min_val = min(Y_test.min(), Y_test_pred.min())
max_val = max(Y_test.max(), Y_test_pred.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    color="red",
    linewidth=2,
    label="Perfect Prediction"
)

plt.title("Predicted vs Actual")
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("images/predicted_vs_actual.png")
plt.close()

print("\nModel Saved Successfully!")

print(f"\nTotal Iterations Trained: {len(cost_history):,}")

print("\nWeights:")
print(weights)

print("\nBias:")
print(bias)