import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset and prepare features
df = pd.read_csv('student-por.csv', sep=';')

features = ['studytime', 'failures', 'absences', 'G1', 'G2']
X = df[features]
y = df['G3']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Iterate across degrees 1 to 5
degrees = list(range(1, 6))
complexity_results = []

for d in degrees:
    # a. Transform features
    poly = PolynomialFeatures(degree=d, include_bias=False)
    X_tr_poly = poly.fit_transform(X_train)
    X_te_poly = poly.transform(X_test)
    
    # b. Train regression model
    model = LinearRegression()
    model.fit(X_tr_poly, y_train)
    
    # Predictions
    y_tr_pred = model.predict(X_tr_poly)
    y_te_pred = model.predict(X_te_poly)
    
    # c. Calculate errors
    tr_mse = mean_squared_error(y_train, y_tr_pred)
    te_mse = mean_squared_error(y_test, y_te_pred)
    tr_rmse = np.sqrt(tr_mse)
    te_rmse = np.sqrt(te_mse)
    
    complexity_results.append({
        'Degree': d,
        'Num_Features': X_tr_poly.shape[1],
        'Train_MSE': tr_mse,
        'Test_MSE': te_mse,
        'Train_RMSE': tr_rmse,
        'Test_RMSE': te_rmse
    })

# d. Store results in a DataFrame
df_complexity = pd.DataFrame(complexity_results)
print("=== Model Complexity Metrics ===")
print(df_complexity[['Degree', 'Train_MSE', 'Test_MSE', 'Train_RMSE', 'Test_RMSE']].to_string(index=False))

# e. Plot Polynomial Degree vs Training & Testing RMSE
plt.figure(figsize=(9, 5))
plt.plot(df_complexity['Degree'], df_complexity['Train_RMSE'], marker='o', color='royalblue', label='Training RMSE')
plt.plot(df_complexity['Degree'], df_complexity['Test_RMSE'], marker='s', color='crimson', label='Testing RMSE')
plt.title('Polynomial Degree vs. Training & Testing RMSE')
plt.xlabel('Polynomial Degree')
plt.ylabel('RMSE (Log Scale)')
plt.yscale('log')
plt.xticks(degrees)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()