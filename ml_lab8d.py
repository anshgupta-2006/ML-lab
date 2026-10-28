from sklearn.datasets import load_digits
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 1. Load dataset and split
digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
performance_data = []

# 2. Train and evaluate each kernel
for k in kernels:
  clf = SVC(kernel=k)
  clf.fit(X_train, y_train)

  y_train_pred = clf.predict(X_train)
  y_test_pred = clf.predict(X_test)

  performance_data.append({
      'Kernel': k,
      'Train Acc': accuracy_score(y_train, y_train_pred),
      'Test Acc': accuracy_score(y_test, y_test_pred),
      'Precision': precision_score(y_test, y_test_pred, average='weighted'),
      'Recall': recall_score(y_test, y_test_pred, average='weighted'),
      'F1-Score': f1_score(y_test, y_test_pred, average='weighted'),
  })

# 3. Print the comparison table
print(
    f"{'Kernel':<10} | {'Train Acc':<10} | {'Test Acc':<10} |"
    f" {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}"
)
print("-" * 72)
for row in performance_data:
  print(
      f"{row['Kernel']:<10} | {row['Train Acc']:<10.4f} |"
      f" {row['Test Acc']:<10.4f} | {row['Precision']:<10.4f} |"
      f" {row['Recall']:<10.4f} | {row['F1-Score']:<10.4f}"
  )

# 4. Identify best performing kernel
best_model = max(performance_data, key=lambda x: x["Test Acc"])
print(
    f"\nBest-Performing Kernel: {best_model['Kernel'].upper()} with a Test"
    f" Accuracy of {best_model['Test Acc']:.4f}"
)