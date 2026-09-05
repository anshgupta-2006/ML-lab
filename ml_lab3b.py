import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load Student Performance Dataset (Math Course)
df = pd.read_csv('student-mat.csv', sep=";")

# (a) Take the complete dataset
data = df.copy()

# (b) Convert categorical columns to numerical using LabelEncoder
label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Separate features and target (predicting final grade G3)
X_full = data.drop(columns=['G3'])
y_full = data['G3']

# (c) Scale the dataset using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_full)

# (d) Split into train and test sets (70:30 ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_full, test_size=0.30, random_state=42
)

# (e) Fit Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- Scikit-Learn Model Evaluation ---")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE):       {mae:.4f}")
print(f"R² Score:                        {r2:.4f}")