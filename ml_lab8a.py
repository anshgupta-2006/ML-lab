import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load the dataset
digits = load_digits()

# --- a. Check the shape, samples, features, and target classes ---
print("Dataset Shape (Data matrix):", digits.data.shape)
print("Number of Samples:", digits.data.shape[0])
print("Number of Input Features:", digits.data.shape[1])
print("Number of Target Classes:", len(set(digits.target)))

# --- b. Display the first 10 handwritten digit images ---
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
  ax.imshow(digits.images[i], cmap='gray')
  ax.set_title(
      f"Label: {digits.target[i]}"
  )
  ax.axis('off')
plt.tight_layout()
plt.show()

# --- c. Extract independent and dependent variables ---
X = digits.data  # Independent variables (features)
y = digits.target  # Dependent variable (target labels)

# --- d. Split into 80% training and 20% testing data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"Training set shape (X_train): {X_train.shape}")
print(f"Testing set shape (X_test): {X_test.shape}")