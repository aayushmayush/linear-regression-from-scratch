import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -----------------------------
# Load Dataset
# -----------------------------
insurance_data = pd.read_csv("../data/insurance.csv")

X = insurance_data[["age", "sex", "bmi", "children", "smoker", "region"]]
Y = insurance_data["charges"]

# Convert categorical columns
X = pd.get_dummies(
    X,
    columns=["sex", "smoker", "region"],
    drop_first=True
)

# Convert everything to float
X = X.astype(np.float64)

# -----------------------------
# Same Train/Test Split
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
# Load Your Model
# -----------------------------
weights = np.load("weights.npy")
bias = np.load("bias.npy")[0]

# -----------------------------
# Predictions using YOUR model
# -----------------------------
my_train_pred = X_train @ weights + bias
my_test_pred = X_test @ weights + bias

# -----------------------------
# Train Scikit-learn Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, Y_train)

sk_train_pred = model.predict(X_train)
sk_test_pred = model.predict(X_test)

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

print("\nWeights:")
print(weights)

print("\nBias:")
print(bias)

print("\nTraining Cost:")
print(cost(Y_train, my_train_pred))

print("\nTesting Cost:")
print(cost(Y_test, my_test_pred))

print("\n==============================")
print("SCIKIT-LEARN")
print("==============================")

print("\nWeights:")
print(model.coef_)

print("\nBias:")
print(model.intercept_)

print("\nTraining Cost:")
print(cost(Y_train, sk_train_pred))

print("\nTesting Cost:")
print(cost(Y_test, sk_test_pred))

print("\n==============================")
print("DIFFERENCE")
print("==============================")

print("\nWeight Difference:")
print(weights - model.coef_)

print("\nBias Difference:")
print(bias - model.intercept_)

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