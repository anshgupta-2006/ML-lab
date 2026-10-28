import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Set random seed for reproducibility
SEED = 42

# ==========================================
# a. Read Dataset, EDA & Preprocessing
# ==========================================
# Load dataset (adjust path as necessary)
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('Carseats.csv')

# 1. Replace empty strings or whitespace-only values with NaN
df = df.replace(r'^\s*$', np.nan, regex=True)

print("--- Missing Values After Cleaning Empty Strings ---")
print(df.isnull().sum())

# 2. Handle missing values for numeric columns (e.g., fill with median)
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# For categorical columns with missing values, you can fill with the mode (most frequent)
categorical_cols = ['ShelveLoc', 'Urban', 'US']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

# 3. Now proceed with encoding categorical variables
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 4. Create binary target 'High'
df_encoded['High'] = np.where(df_encoded['Sales'] > 8, 1, 0)

# Define Feature Matrix (X) and Target Vector (Y)
X = df_encoded.drop(columns=['Sales', 'High'])
Y = df_encoded['High']

# ==========================================
# b. Target Variable Creation & Feature Matrix
# ==========================================
# Create binary target 'High': 1 if Sales > 8, else 0
df_encoded['High'] = np.where(df_encoded['Sales'] > 8, 1, 0)

# Define Feature Matrix (X) and Target Vector (Y)
X = df_encoded.drop(columns=['Sales', 'High'])
Y = df_encoded['High']

print(f"\nTarget Class Distribution:\n{Y.value_counts(normalize=True)}")

# ==========================================
# c. Train-Test Split (80/20)
# ==========================================
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=SEED, stratify=Y
)

# ==========================================
# d. Train Decision Tree & Random Forest Models
# ==========================================
# Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=SEED)
dt_model.fit(X_train, Y_train)

# Default Random Forest Classifier
rf_model = RandomForestClassifier(random_state=SEED)
rf_model.fit(X_train, Y_train)

# ==========================================
# e. Model Evaluation Function
# ==========================================
def evaluate_model(model, X_tr, Y_tr, X_te, Y_te, model_name="Model"):
    tr_preds = model.predict(X_tr)
    te_preds = model.predict(X_te)
    
    metrics = {
        'Train Acc': accuracy_score(Y_tr, tr_preds),
        'Test Acc': accuracy_score(Y_te, te_preds),
        'Precision': precision_score(Y_te, te_preds),
        'Recall': recall_score(Y_te, te_preds),
        'F1-Score': f1_score(Y_te, te_preds)
    }
    
    print(f"=== Evaluation: {model_name} ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    print("\nTest Confusion Matrix:")
    print(confusion_matrix(Y_te, te_preds))
    print("-" * 40)
    return metrics

dt_metrics = evaluate_model(dt_model, X_train, Y_train, X_test, Y_test, "Decision Tree (Default)")
rf_metrics = evaluate_model(rf_model, X_train, Y_train, X_test, Y_test, "Random Forest (Default)")

# ==========================================
# f. Hyperparameter Tuning for Random Forest
# ==========================================
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'max_features': ['sqrt', 'log2', None]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=SEED),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, Y_train)

best_rf = grid_search.best_estimator_
print("\nBest Hyperparameters found by GridSearchCV:")
print(grid_search.best_params_)

best_rf_metrics = evaluate_model(best_rf, X_train, Y_train, X_test, Y_test, "Random Forest (Tuned)")

# ==========================================
# h. Feature Importance Scores & Visualization
# ==========================================
importances = best_rf.feature_importances_
feature_names = X.columns
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
plt.title('Random Forest - Feature Importance Scores')
plt.xlabel('Importance Score')
plt.ylabel('Attributes')
plt.tight_layout()
plt.show()