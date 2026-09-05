import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# a & b. Load dataset
df = pd.read_csv('student-por.csv', sep=';')

# c & d. Select numerical features and target variable
features = ['studytime', 'failures', 'absences', 'G1', 'G2']
X = df[features]
y = df['G3']

# e. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# f. Polynomial transformation (Degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# g. Train Linear Regression model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# h. Evaluate model metrics
y_pred = model.predict(X_test_poly)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("=== Degree 2 Polynomial Regression Evaluation ===")
print(f"Mean Squared Error (MSE) : {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score                      : {r2:.4f}\n")

# i. Display actual vs predicted values
results_df = pd.DataFrame({
    'Actual G3': y_test.values[:10],
    'Predicted G3': np.round(y_pred[:10], 2)
})
print("Actual vs Predicted (First 10 test samples):")
print(results_df.to_string(index=False))