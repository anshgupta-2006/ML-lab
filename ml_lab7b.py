import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# ---------------------------------------------------------
# Part a: Load Dataset & Separate by Column Position
# ---------------------------------------------------------
# Load raw CSV without assuming column names
df = pd.read_csv("pima-indians-diabetes.csv", header=None)

# If the first row contains text headers (like 'Pregnancies'), drop it
if not str(df.iloc[0, 0]).replace('.', '', 1).isdigit():
    df = df.iloc[1:].reset_index(drop=True)

# Ensure all columns are numeric floats
df = df.apply(pd.to_numeric, errors='coerce').dropna()

# BULLETPROOF SLICING: Use column positions instead of string names
X = df.iloc[:, :-1]        # Columns 0 to 7 (Features)
y = df.iloc[:, -1].astype(int)  # Column 8 (Target / Outcome)

# Feature column names for readability
feature_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]
X.columns = feature_names

# 80/20 Train-Test Split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

def get_metrics(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0)
    }

# ---------------------------------------------------------
# Part b: Evaluate Across Different Depths
# ---------------------------------------------------------
depths = [2, 3, 4, 5, 7, 10, None]
depth_results = []

for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    
    train_m = get_metrics(y_train, clf.predict(X_train))
    test_m = get_metrics(y_test, clf.predict(X_test))
    
    depth_results.append({
        'max_depth': str(depth),
        'Train Acc': round(train_m['Accuracy'], 4),
        'Test Acc': round(test_m['Accuracy'], 4),
        'Train F1': round(train_m['F1-Score'], 4),
        'Test F1': round(test_m['F1-Score'], 4)
    })

depth_df = pd.DataFrame(depth_results)

print("\n=== Performance Across Tree Depths ===")
print(depth_df.to_string(index=False))

# ---------------------------------------------------------
# Part d & e: Cost-Complexity Pruning (ccp_alpha)
# ---------------------------------------------------------
base_clf = DecisionTreeClassifier(random_state=42)
path = base_clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = [a for a in path.ccp_alphas if a >= 0][:15]

prune_results = []
prune_clfs = []

for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    clf.fit(X_train, y_train)
    prune_clfs.append(clf)
    
    train_m = get_metrics(y_train, clf.predict(X_train))
    test_m = get_metrics(y_test, clf.predict(X_test))
    
    prune_results.append({
        'ccp_alpha': round(alpha, 5),
        'Train Acc': round(train_m['Accuracy'], 4),
        'Test Acc': round(test_m['Accuracy'], 4),
        'Train F1': round(train_m['F1-Score'], 4),
        'Test F1': round(test_m['F1-Score'], 4)
    })

prune_df = pd.DataFrame(prune_results)

best_prune_idx = prune_df['Test F1'].idxmax()
best_alpha = prune_df.loc[best_prune_idx, 'ccp_alpha']

print("\n=== Best Pruned Tree Model ===")
print(f"Optimal ccp_alpha: {best_alpha:.5f}")
print(f"Test Accuracy:    {prune_df.loc[best_prune_idx, 'Test Acc']:.4f}")
print(f"Test F1-Score:    {prune_df.loc[best_prune_idx, 'Test F1']:.4f}")

# ---------------------------------------------------------
# Part f: Visualizations
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Depth vs Accuracy
axes[0].plot(depth_df['max_depth'], depth_df['Train Acc'], marker='o', label='Train Acc')
axes[0].plot(depth_df['max_depth'], depth_df['Test Acc'], marker='s', label='Test Acc')
axes[0].set_title('Tree Depth vs Accuracy')
axes[0].set_xlabel('max_depth')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

# Plot 2: Pruning vs Accuracy
axes[1].plot(prune_df['ccp_alpha'], prune_df['Train Acc'], marker='o', label='Train Acc')
axes[1].plot(prune_df['ccp_alpha'], prune_df['Test Acc'], marker='s', label='Test Acc')
axes[1].axvline(x=best_alpha, color='red', linestyle='--', label=f'Best alpha ({best_alpha})')
axes[1].set_title('Cost-Complexity Pruning vs Accuracy')
axes[1].set_xlabel('ccp_alpha')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.show()