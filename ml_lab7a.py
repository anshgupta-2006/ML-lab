import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

# Define evaluation helper function at the top level
def evaluate_model(y_true, y_pred, name):
    return {
        'Model': name,
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'F1-Score': round(f1_score(y_true, y_pred, zero_division=0), 4)
    }

# ---------------------------------------------------------
# Part a: Read Dataset & Clean
# ---------------------------------------------------------
df = pd.read_csv("Carseats.csv")

# Drop extra trailing column if present
if 'Unnamed: 11' in df.columns:
    df = df.drop(columns=['Unnamed: 11'])

print(f"Cleaned Dataset Shape: {df.shape}")

# ---------------------------------------------------------
# Part b: Target & Feature Definition
# ---------------------------------------------------------
# High Sales = 1 if Sales > 8 else 0
df['High'] = (df['Sales'] > 8).astype(int)

# ---------------------------------------------------------
# Part e (EDA): Class Distribution & Correlations
# ---------------------------------------------------------
print("\n--- Target Class Proportion ---")
print(df['High'].value_counts(normalize=True))

numeric_cols = df.select_dtypes(include=[np.number]).columns
print("\n--- Correlations with Target (High) ---")
print(df[numeric_cols].corr()['High'].sort_values(ascending=False))

# ---------------------------------------------------------
# Parts c & d: Baseline Gini vs. Entropy
# ---------------------------------------------------------
X_raw = df.drop(columns=['Sales', 'High'])
Y = df['High']

# One-hot encoding for categorical variables
X_encoded = pd.get_dummies(X_raw, columns=['ShelveLoc', 'Urban', 'US'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, Y, test_size=0.2, random_state=42, stratify=Y
)

# Gini Model
dt_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
dt_gini.fit(X_train, y_train)
y_pred_gini = dt_gini.predict(X_test)

# Entropy Model
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_entropy.fit(X_train, y_train)
y_pred_entropy = dt_entropy.predict(X_test)

baseline_results = pd.DataFrame([
    evaluate_model(y_test, y_pred_gini, 'Baseline Gini'),
    evaluate_model(y_test, y_pred_entropy, 'Baseline Entropy')
])

print("\n--- Baseline Model Comparison ---")
print(baseline_results.to_string(index=False))

# ---------------------------------------------------------
# Part e: Preprocessing & Hyperparameter Optimization
# ---------------------------------------------------------
df_prep = df.copy()

# Ordinal encoding for ShelveLoc (Bad < Medium < Good)
shelve_map = {'Bad': 0, 'Medium': 1, 'Good': 2}
df_prep['ShelveLoc'] = df_prep['ShelveLoc'].map(shelve_map)

# Binary encoding
df_prep['Urban'] = (df_prep['Urban'] == 'Yes').astype(int)
df_prep['US'] = (df_prep['US'] == 'Yes').astype(int)

# Feature engineering
df_prep['Price_Diff'] = df_prep['CompPrice'] - df_prep['Price']

X_opt = df_prep.drop(columns=['Sales', 'High'])
Y_opt = df_prep['High']

X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(
    X_opt, Y_opt, test_size=0.2, random_state=42, stratify=Y_opt
)

# Grid search tuning
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 4, 5, 6, 8],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'ccp_alpha': [0.0, 0.005, 0.01, 0.015]
}

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train_opt, y_train_opt)
best_dt = grid_search.best_estimator_

# Evaluate optimized model using the top-level evaluate_model function
y_pred_opt = best_dt.predict(X_test_opt)
opt_metrics = evaluate_model(y_test_opt, y_pred_opt, 'Optimized Decision Tree')

print("\n--- Optimized Model Evaluation ---")
print(pd.DataFrame([opt_metrics]).to_string(index=False))
print("\nBest Hyperparameters:", grid_search.best_params_)