import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# Set random seed for reproducibility
SEED = 42

# ==========================================
# a. Load Dataset, EDA & Preprocessing
# ==========================================
# Load dataset (adjust path as necessary)
df = pd.read_csv('pima-indians-diabetes.csv')

print("--- Initial Data Overview ---")
print(df.info())
print("\n--- Missing Values Check (Before Zero Handling) ---")
print(df.isnull().sum())

# Handle invalid zero values in columns where 0 is biologically impossible
zero_invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_invalid_cols:
    df[col] = df[col].replace(0, np.nan)

print("\n--- Missing Values After Replacing Invalid Zeros ---")
print(df.isnull().sum())

# Impute missing values using the median of each column
for col in zero_invalid_cols:
    df[col] = df[col].fillna(df[col].median())

# ==========================================
# b. Feature Selection & Stratified Split
# ==========================================
# 1. Strip any leading/trailing spaces from column names FIRST
df.columns = df.columns.str.strip()

# 2. Print columns to verify the exact name of the target column
print("Available columns:", df.columns.tolist())

# Automatically find the target column case-insensitively ('outcome', 'class', etc.)
target_column = None
for col in df.columns:
    if col.lower() in ['outcome', 'class', 'target']:
        target_column = col
        break

# Fallback: if none of the names match, pick the last column in the dataset
if target_column is None:
    target_column = df.columns[-1]

print(f"Selected target column: {target_column}")

# Now safely drop and assign
X = df.drop(columns=[target_column])
Y = df[target_column]

print(f"\nTarget Class Distribution:\n{Y.value_counts(normalize=True)}")

# Split dataset into 80% training and 20% testing data with stratification
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=SEED, stratify=Y
)

# ==========================================
# c & d. Train and Evaluate Four Models
# ==========================================
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=SEED),
    'Random Forest': RandomForestClassifier(random_state=SEED),
    'AdaBoost': AdaBoostClassifier(random_state=SEED),
    'Gradient Boosting': GradientBoostingClassifier(random_state=SEED)
}

evaluation_results = []

for name, model in models.items():
    model.fit(X_train, Y_train)
    tr_preds = model.predict(X_train)
    te_preds = model.predict(X_test)
    
    metrics = {
        'Model': name,
        'Train Acc': accuracy_score(Y_train, tr_preds),
        'Test Acc': accuracy_score(Y_test, te_preds),
        'Precision': precision_score(Y_test, te_preds),
        'Recall': recall_score(Y_test, te_preds),
        'F1-Score': f1_score(Y_test, te_preds)
    }
    evaluation_results.append(metrics)
    
    print(f"=== {name} Confusion Matrix ===")
    print(confusion_matrix(Y_test, te_preds))
    print("-" * 40)

results_df = pd.DataFrame(evaluation_results)
print("\n--- Model Performance Comparison Table ---")
print(results_df.to_string(index=False))

# ==========================================
# e. Experiment with n_estimators (RF & AdaBoost)
# ==========================================
n_estimators_list = [10, 50, 100, 200]
rf_n_est_results, ada_n_est_results = [], []

for n in n_estimators_list:
    # Random Forest
    rf = RandomForestClassifier(n_estimators=n, random_state=SEED)
    rf.fit(X_train, Y_train)
    rf_n_est_results.append({
        'n_estimators': n,
        'Train F1': f1_score(Y_train, rf.predict(X_train)),
        'Test F1': f1_score(Y_test, rf.predict(X_test))
    })
    
    # AdaBoost
    ada = AdaBoostClassifier(n_estimators=n, random_state=SEED)
    ada.fit(X_train, Y_train)
    ada_n_est_results.append({
        'n_estimators': n,
        'Train F1': f1_score(Y_train, ada.predict(X_train)),
        'Test F1': f1_score(Y_test, ada.predict(X_test))
    })

# ==========================================
# f. Experiment with learning_rate (AdaBoost)
# ==========================================
learning_rates = [0.01, 0.1, 0.5, 1.0]
ada_lr_results = []

for lr in learning_rates:
    ada = AdaBoostClassifier(learning_rate=lr, random_state=SEED)
    ada.fit(X_train, Y_train)
    ada_lr_results.append({
        'learning_rate': lr,
        'Train F1': f1_score(Y_train, ada.predict(X_train)),
        'Test F1': f1_score(Y_test, ada.predict(X_test))
    })

# ==========================================
# g. Plotting Relationships
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: RF n_estimators vs F1
rf_df = pd.DataFrame(rf_n_est_results)
axes[0].plot(rf_df['n_estimators'], rf_df['Train F1'], marker='o', label='Train F1')
axes[0].plot(rf_df['n_estimators'], rf_df['Test F1'], marker='s', label='Test F1')
axes[0].set_title('Random Forest: n_estimators vs F1')
axes[0].set_xlabel('n_estimators')
axes[0].set_ylabel('F1-Score')
axes[0].legend()
axes[0].grid(True)

# Plot 2: AdaBoost n_estimators vs F1
ada_n_df = pd.DataFrame(ada_n_est_results)
axes[1].plot(ada_n_df['n_estimators'], ada_n_df['Train F1'], marker='o', label='Train F1')
axes[1].plot(ada_n_df['n_estimators'], ada_n_df['Test F1'], marker='s', label='Test F1')
axes[1].set_title('AdaBoost: n_estimators vs F1')
axes[1].set_xlabel('n_estimators')
axes[1].set_ylabel('F1-Score')
axes[1].legend()
axes[1].grid(True)

# Plot 3: AdaBoost learning_rate vs F1
ada_lr_df = pd.DataFrame(ada_lr_results)
axes[2].plot(ada_lr_df['learning_rate'], ada_lr_df['Train F1'], marker='o', label='Train F1')
axes[2].plot(ada_lr_df['learning_rate'], ada_lr_df['Test F1'], marker='s', label='Test F1')
axes[2].set_title('AdaBoost: learning_rate vs F1')
axes[2].set_xlabel('Learning Rate')
axes[2].set_ylabel('F1-Score')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()