import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -----------------------------
# Load Dataset
# -----------------------------
insurance_data = pd.read_csv("../data/insurance.csv")

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

# sklearn expects 2D input
X_train_2d = X_train.reshape(-1, 1)
X_test_2d = X_test.reshape(-1, 1)

# -----------------------------
# Load Your Model
# -----------------------------
m = np.load("weights.npy")[0]
c = np.load("bias.npy")[0]

# -----------------------------
# Predictions
# -----------------------------
my_train_pred = m * X_train + c
my_test_pred = m * X_test + c

# -----------------------------
# Train Scikit-learn Model
# -----------------------------
model = LinearRegression()
model.fit(X_train_2d, Y_train)

sk_train_pred = model.predict(X_train_2d)
sk_test_pred = model.predict(X_test_2d)

# -----------------------------
# Cost Function
# -----------------------------
def cost(y_true, y_pred):
    return np.sum((y_pred - y_true) ** 2) / (2 * len(y_true))

# -----------------------------
# Print Results
# -----------------------------
print("\n==============================")
print("YOUR MODEL")
print("==============================")

print("\nSlope (m):")
print(m)

print("\nIntercept (c):")
print(c)

print("\nTraining Cost:")
print(cost(Y_train, my_train_pred))

print("\nTesting Cost:")
print(cost(Y_test, my_test_pred))

print("\n==============================")
print("SCIKIT-LEARN")
print("==============================")

print("\nSlope (m):")
print(model.coef_[0])

print("\nIntercept (c):")
print(model.intercept_)

print("\nTraining Cost:")
print(cost(Y_train, sk_train_pred))

print("\nTesting Cost:")
print(cost(Y_test, sk_test_pred))

print("\n==============================")
print("DIFFERENCE")
print("==============================")

print("\nSlope Difference:")
print(m - model.coef_[0])

print("\nIntercept Difference:")
print(c - model.intercept_)

print("\n==============================")
print("FIRST 10 TEST PREDICTIONS")
print("==============================")

for i in range(10):
    print(
        f"{i+1:2d}. "
        f"Mine: {my_test_pred[i]:10.2f} | "
        f"Sklearn: {sk_test_pred[i]:10.2f} | "
        f"Actual: {Y_test[i]:10.2f}"
    )