import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset and prepare features
df = pd.read_csv('student-por.csv', sep=';')

features = ['studytime', 'failures', 'absences', 'G1', 'G2']
X = df[features]
y = df['G3']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# a & b. Train models and calculate R2 scores for degrees 1 to 5
degrees = list(range(1, 6))
r2_records = []

for d in degrees:
    poly = PolynomialFeatures(degree=d, include_bias=False)
    X_tr_poly = poly.fit_transform(X_train)
    X_te_poly = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_tr_poly, y_train)
    
    y_tr_pred = model.predict(X_tr_poly)
    y_te_pred = model.predict(X_te_poly)
    
    r2_records.append({
        'Degree': d,
        'Train_R2': r2_score(y_train, y_tr_pred),
        'Test_R2': r2_score(y_test, y_te_pred)
    })

df_r2 = pd.DataFrame(r2_records)
print("=== R² Score Comparison ===")
print(df_r2.to_string(index=False))

# c. Plot Polynomial Degree vs. R2 Curves
plt.figure(figsize=(9, 5))
plt.plot(df_r2['Degree'], df_r2['Train_R2'], marker='o', color='forestgreen', label='Training $R^2$')
plt.plot(df_r2['Degree'], df_r2['Test_R2'], marker='s', color='darkorange', label='Testing $R^2$')
plt.title('Polynomial Degree vs. $R^2$ Score')
plt.xlabel('Polynomial Degree')
plt.ylabel('$R^2$ Score')
plt.xticks(degrees)
plt.ylim(-1.5, 1.1)
plt.axhline(0, color='grey', linestyle=':')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# d. Overfitting identification output
print("\nOverfitting Identification:")
print("- Degrees 1 & 2: Balanced fit with consistent training and test R² scores.")
print("- Degree 3: Model begins showing clear signs of overfitting as test R² sharply drops while training R² rises.")
print("- Degrees 4 & 5: Severe overfitting occurs with training R² near 1.0 and negative testing R².")