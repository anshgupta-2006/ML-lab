import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_auc_score
)

# --- Steps a & b: Load Dataset & Preprocess ---
url = "https://raw.githubusercontent.com/madmashup/targeted-marketing-predictive-engine/master/banking.csv"
df = pd.read_csv(url)

# Check for missing values
print("=== Missing Value Check ===")
print(f"Total missing values: {df.isnull().sum().sum()}\n")

# Separate target variable 'y' and feature matrix 'X'
X = df.drop(columns=['y'])
y = df['y']

# One-Hot Encode categorical variables and scale numerical features
X_encoded = pd.get_dummies(X, drop_first=True)

# --- Steps c through l: Model Training, Thresholding & Evaluation ---
split_ratios = {'70:30 Split': 0.30, '80:20 Split': 0.20}
thresholds = [0.3, 0.5, 0.7]

evaluation_results = []
models = {}
scaler = StandardScaler()

# Prepare 2x3 subplot matrix for Confusion Matrices
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Confusion Matrices Across Train/Test Splits and Probability Thresholds", fontsize=14, fontweight='bold')

for row_idx, (split_name, test_sz) in enumerate(split_ratios.items()):
    # Step c, d, e: Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_sz, random_state=42, stratify=y
    )
    
    # Feature Scaling
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step f: Fit Logistic Regression Model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    models[split_name] = (model, X_encoded.columns)
    
    # Step g: Regression Coefficients
    coef_df = pd.DataFrame({'Feature': X_encoded.columns, 'Coefficient': model.coef_[0]})
    top_pos_coefs = coef_df.sort_values(by='Coefficient', ascending=False).head(3)
    top_neg_coefs = coef_df.sort_values(by='Coefficient', ascending=True).head(3)
    print(f"=== {split_name} - Top Positive Coefficients ===")
    print(top_pos_coefs.to_string(index=False))
    print(f"\n=== {split_name} - Top Negative Coefficients ===")
    print(top_neg_coefs.to_string(index=False), "\n" + "-"*50)
    
    # Step h & i: Predictions and Probabilities
    y_probs = model.predict_proba(X_test_scaled)[:, 1]
    
    # Step j & k: Metric Evaluation across Thresholds
    for col_idx, thresh in enumerate(thresholds):
        y_pred_thresh = (y_probs >= thresh).astype(int)
        
        acc = accuracy_score(y_test, y_pred_thresh)
        prec = precision_score(y_test, y_pred_thresh, zero_division=0)
        rec = recall_score(y_test, y_pred_thresh)
        f1 = f1_score(y_test, y_pred_thresh)
        
        evaluation_results.append({
            'Split Ratio': split_name,
            'Threshold': thresh,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4)
        })
        
        # Step l: Plot Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_thresh)
        ax = axes[row_idx, col_idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_title(f"{split_name} | Threshold: {thresh}")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")

plt.tight_layout()
plt.show()

# Display Comparison Table
results_df = pd.DataFrame(evaluation_results)
print("=== Classification Metrics Comparison Table ===")
print(results_df.to_string(index=False))

# --- Step m: Single Feature Evaluation (Independent Variable vs Dependent Variable Y) ---
print("\n" + "="*50)
print("=== Single Independent Variable Analysis (vs Target Y) ===")

single_feature_performance = []

for col in X.columns:
    # Prepare feature sub-matrix (encode if categorical, scale if numerical)
    X_single = df[[col]]
    if X_single[col].dtype == 'object':
        X_single = pd.get_dummies(X_single, drop_first=True)
    
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_single, y, test_size=0.20, random_state=42, stratify=y
    )
    
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)
    
    lr_single = LogisticRegression(max_iter=1000, random_state=42)
    lr_single.fit(X_tr_sc, y_tr)
    
    y_single_probs = lr_single.predict_proba(X_te_sc)[:, 1]
    y_single_preds = lr_single.predict(X_te_sc)
    
    auc = roc_auc_score(y_te, y_single_probs)
    f1_single = f1_score(y_te, y_single_preds, zero_division=0)
    acc_single = accuracy_score(y_te, y_single_preds)
    
    single_feature_performance.append({
        'Feature': col,
        'ROC-AUC': round(auc, 4),
        'F1-Score': round(f1_single, 4),
        'Accuracy': round(acc_single, 4)
    })

single_df = pd.DataFrame(single_feature_performance).sort_values(by='ROC-AUC', ascending=False)
print(single_df.to_string(index=False))

top_feature = single_df.iloc[0]['Feature']
top_auc = single_df.iloc[0]['ROC-AUC']
print(f"\nMost Desirable Single Feature: '{top_feature}' (ROC-AUC: {top_auc})")