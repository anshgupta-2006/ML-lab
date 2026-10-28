from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load dataset and split
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# --- a. Polynomial Degree Analysis ---
print('--- A. Polynomial Degree Analysis (kernel="poly") ---')
for deg in [2, 3, 4, 5]:
  clf = SVC(kernel='poly', degree=deg)
  clf.fit(X_train, y_train)
  train_acc = accuracy_score(y_train, clf.predict(X_train))
  test_acc = accuracy_score(y_test, clf.predict(X_test))
  print(
      f"Degree: {deg} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}"
  )

# --- b. Gamma Analysis ---
print('\n--- B. Gamma Analysis (kernel="rbf") ---')
for g in [0.001, 0.01, 0.1, 1]:
  clf = SVC(kernel='rbf', gamma=g)
  clf.fit(X_train, y_train)
  train_acc = accuracy_score(y_train, clf.predict(X_train))
  test_acc = accuracy_score(y_test, clf.predict(X_test))
  print(f"Gamma: {g:<5} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

# --- c. C Parameter Analysis ---
print('\n--- C. C Parameter Analysis (kernel="rbf") ---')
for c_val in [0.1, 1, 10, 100]:
  clf = SVC(kernel='rbf', C=c_val)
  clf.fit(X_train, y_train)
  train_acc = accuracy_score(y_train, clf.predict(X_train))
  test_acc = accuracy_score(y_test, clf.predict(X_test))
  print(f"C: {c_val:<6} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")