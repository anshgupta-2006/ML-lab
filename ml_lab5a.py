import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import urllib.request
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('student-por.csv', sep=';')

# --- Steps c & d: Select Numerical Features and Target Variable ---
numerical_features = [
    'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 
    'failures', 'famrel', 'freetime', 'goout', 'Dalc', 
    'Walc', 'health', 'absences', 'G1', 'G2'
]
X = df[numerical_features]
y = df['G3']  # Target variable

# --- Step e: Split Dataset into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features prior to polynomial expansion and regularization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step f: Apply PolynomialFeatures (Degree 2) ---
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# --- Steps g & h: Train Models (Linear, Ridge, Lasso) ---
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression (alpha=10.0)': Ridge(alpha=10.0),
    'Lasso Regression (alpha=0.1)': Lasso(alpha=0.1, max_iter=10000, random_state=42)
}

results = {}
predictions = {}

# --- Step i: Evaluate Models ---
for name, model in models.items():
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)
    predictions[name] = y_pred
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'MSE': mse, 'RMSE': rmse, 'R2 Score': r2}

# --- Step j: Create Comparison Table ---
comparison_df = pd.DataFrame(results).T
print("=== Model Performance Comparison ===")
print(comparison_df.round(4))

# --- Step k: Display Actual vs Predicted Values ---
plt.figure(figsize=(15, 5))

for idx, (name, y_pred) in enumerate(predictions.items(), 1):
    plt.subplot(1, 3, idx)
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue', edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f"{name}")
    plt.xlabel("Actual G3 Grades")
    plt.ylabel("Predicted G3 Grades")
    plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()