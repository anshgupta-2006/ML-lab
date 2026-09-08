import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, Binarizer
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)

columns = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

# header=0 replaces the existing text header row instead of loading it as data
df = pd.read_csv("pima-indians-diabetes.csv", names=columns, header=0)

# Convert all column values explicitly to numeric floats
df = df.astype(float)

# Separate features (X) and label (y)
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# ==========================================
# a) Feature Rescaling (MinMax 0 to 1)
# ==========================================
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ==========================================
# b) Binarizer Transformation (Threshold = 0.0)
# ==========================================
binarizer = Binarizer(threshold=0.0)
X_binarized = pd.DataFrame(binarizer.fit_transform(X), columns=X.columns)

print("--- Head of Binarized Features (Threshold = 0.0) ---")
print(X_binarized.head())
print("\n")

# ==========================================
# c) K-Fold Cross Validation with KNN
# ==========================================
kf = KFold(n_splits=10, shuffle=True, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

# Predictions on scaled features using 10-fold CV
y_pred_scaled = cross_val_predict(knn, X_scaled, y, cv=kf)

# ==========================================
# d) Confusion Matrix Plot
# ==========================================
cm = confusion_matrix(y, y_pred_scaled)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Diabetic', 'Diabetic'])

# Explicitly create figure and axes, then pass ax to disp.plot()
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax)

plt.title("KNN Confusion Matrix (Scaled Features)")
plt.show()

# ==========================================
# e) Performance Evaluation (Before vs After Scaling)
# ==========================================
# Predictions on raw (unscaled) features
y_pred_raw = cross_val_predict(knn, X, y, cv=kf)

def compute_metrics(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred)
    }

metrics_raw = compute_metrics(y, y_pred_raw)
metrics_scaled = compute_metrics(y, y_pred_scaled)

results_df = pd.DataFrame([metrics_raw, metrics_scaled], index=['Before Scaling (Raw)', 'After Scaling (Scaled)'])
print("--- Performance Comparison ---")
print(results_df.round(4))