from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 1. Load dataset and split into training and testing sets
digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 2. Define the kernels to test
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
results = {}

print("--- SVM Kernel Performance Comparison ---")
for kernel_type in kernels:
  # Initialize and train SVC with the specific kernel
  svm_model = SVC(kernel=kernel_type)
  svm_model.fit(X_train, y_train)

  # Predict and evaluate accuracy
  y_pred = svm_model.predict(X_test)
  acc = accuracy_score(y_test, y_pred)
  results[kernel_type] = acc

  print(f"Kernel: {kernel_type:<10} | Test Accuracy: {acc:.4f}")

# 3. Identify the best performing kernel
best_kernel = max(results, key=results.get)
print(
    f"\nBest Performing Kernel: {best_kernel.upper()} with an accuracy of"
    f" {results[best_kernel]:.4f}"
)