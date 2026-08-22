import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load Student Performance Dataset (Math Course)
url = "https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/Students_Academic_Performance/student-mat.csv"
df = pd.read_csv(url, sep=";")

# (a) Extract G2 (X) and G3 (Y)
X = df[['G2']].values
Y = df[['G3']].values

# (b) Statistical values using describe() and shapes
print("--- Statistical Summary of X (G2) ---")
print(df[['G2']].describe())

print("\n--- Statistical Summary of Y (G3) ---")
print(df[['G3']].describe())

print(f"\nShape of X: {X.shape}")
print(f"Shape of Y: {Y.shape}")

# (c) Prediction function
def predict(X, w, b):
    return (w * X) + b

# (d) Mean Squared Error (MSE) Loss function
def compute_loss(Y_pred, Y_true):
    N = len(Y_true)
    return np.mean((Y_pred - Y_true) ** 2)

# (e) Gradient computation function
def compute_gradients(X, Y_true, Y_pred):
    N = len(Y_true)
    dw = (2 / N) * np.sum((Y_pred - Y_true) * X)
    db = (2 / N) * np.sum(Y_pred - Y_true)
    return dw, db

# Training loop with convergence criterion
def train_linear_regression(X, Y, lr=0.0001, initial_w=0.1, initial_b=0.01, tol=1e-7, max_iters=100000):
    w = initial_w
    b = initial_b
    prev_loss = float('inf')
    loss_history = []

    for i in range(max_iters):
        # Forward pass
        Y_pred = predict(X, w, b)
        loss = compute_loss(Y_pred, Y)
        loss_history.append(loss)

        # Check stopping condition: stop when loss does not improve
        if abs(prev_loss - loss) < tol:
            print(f"Convergence reached at iteration {i+1}.")
            break
        prev_loss = loss

        # Backward pass (gradient descent step)
        dw, db = compute_gradients(X, Y, Y_pred)
        w -= lr * dw
        b -= lr * db

    return w, b, loss_history

# Execute training
weight, bias, losses = train_linear_regression(X, Y, lr=0.0001, initial_w=0.1, initial_b=0.01)

print(f"\nTrained Weight (m): {weight:.4f}")
print(f"Trained Bias (C):   {bias:.4f}")
print(f"Final MSE Loss:     {losses[-1]:.4f}")