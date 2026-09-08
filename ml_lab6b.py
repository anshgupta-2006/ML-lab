import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("card_transdata_fraud.csv")

# -------------------------------------------------------------------
# Step a: Divide features and labels into X and Y
# -------------------------------------------------------------------
X = df[["distance_from_home", "ratio_to_median_purchase_price"]]
Y = df["fraud"]

# -------------------------------------------------------------------
# Step c: Split train and test set into 80:20 ratio
# -------------------------------------------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42, stratify=Y
)

# -------------------------------------------------------------------
# Step i (Part 1): Evaluate KNN WITHOUT Feature Scaling (Baseline)
# -------------------------------------------------------------------
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, Y_train)
Y_pred_unscaled = knn_unscaled.predict(X_test)

# -------------------------------------------------------------------
# Steps b & d: Standardize / Normalize data using StandardScaler
# Standardized to Gaussian distribution: Mean (μ) = 0, Std Dev (σ) = 1
# -------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------------------
# Step e: Fit KNeighborsClassifier on scaled training data
# -------------------------------------------------------------------
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, Y_train)

# -------------------------------------------------------------------
# Step f: Predict X_test
# -------------------------------------------------------------------
Y_pred_scaled = knn_scaled.predict(X_test_scaled)

# -------------------------------------------------------------------
# Steps g & h: Calculate accuracy & performance metrics
# -------------------------------------------------------------------
print("--- SCALED MODEL PERFORMANCE METRICS ---")
print(f"Accuracy : {accuracy_score(Y_test, Y_pred_scaled):.4f}")
print(f"Precision: {precision_score(Y_test, Y_pred_scaled):.4f}")
print(f"Recall   : {recall_score(Y_test, Y_pred_scaled):.4f}")
print(f"F1 Score : {f1_score(Y_test, Y_pred_scaled):.4f}\n")

print("Classification Report:")
print(
    classification_report(
        Y_test, Y_pred_scaled, target_names=["Normal", "Fraud"]
    )
)
print("Confusion Matrix:")
print(confusion_matrix(Y_test, Y_pred_scaled))