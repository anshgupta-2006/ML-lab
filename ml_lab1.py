# 1. Import required libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 2. Load and inspect the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Convert to DataFrame for easier inspection and plotting
df = pd.DataFrame(X, columns=feature_names)
df['species'] = [target_names[i] for i in y]

print("Dataset First 5 Rows:")
print(df.head())
print("\nDataset Summary:")
print(df.describe())

# 3. Plot class distribution and feature visualizations
# Class distribution countplot
plt.figure(figsize=(6, 4))
sns.countplot(x='species', data=df, palette='Set2')
plt.title("Class Distribution in Iris Dataset")
plt.xlabel("Species")
plt.ylabel("Count")
plt.show()

# Pairplot to visualize features across classes
sns.pairplot(df, hue='species', markers=["o", "s", "D"], palette='Set1')
plt.suptitle("Pairplot of Iris Features by Species", y=1.02)
plt.show()

# 4. Split data into training/testing (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}\n")

# 5 & 6. Train KNN model and test with different k values (1, 3, 5, 7)
k_values = [1, 3, 5, 7]
accuracy_results = {}

print("--- Evaluation across different K values ---")
for k in k_values:
    # Initialize classifier
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit model on training data
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)

    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    accuracy_results[k] = acc

    print(f"\nResults for K = {k}:")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# 7 & 8. Plot Accuracy vs. K Value
plt.figure(figsize=(6, 4))
plt.plot(list(accuracy_results.keys()), list(accuracy_results.values()), marker='o', linestyle='--', color='b')
plt.title('KNN Accuracy for Different K Values')
plt.xlabel('K (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.show()