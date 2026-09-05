import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Column names specified in assignment
column_names = [
    'wine_class',
    'Alcohol',
    'Malic acid',
    'Ash',
    'Alcalinity of ash',
    'Magnesium',
    'Total phenols',
    'Flavanoids',
    'Nonflavanoid phenols',
    'Proanthocyanins',
    'Color intensity',
    'Hue',
    'OD280/OD315 of diluted wines',
    'Proline',
]

# 1. Read wine dataset without header and assign column titles
# Adjust file format/name if needed (e.g. 'wine.csv' or 'wine.data')
df_wine = pd.read_csv('wine.csv', header=None, names=column_names)

# 2. Min-Max Normalization on 'Alcohol' column (Scaling to [0, 1])
min_max_scaler = MinMaxScaler()
df_wine['Alcohol_MinMax'] = min_max_scaler.fit_transform(
    df_wine[['Alcohol']]
)

# 3. Plot distribution of Min-Max normalized Alcohol
plt.figure(figsize=(8, 5))
sns.histplot(df_wine['Alcohol_MinMax'], kde=True, color='purple', bins=15)
plt.title('Distribution of Min-Max Normalized Alcohol')
plt.xlabel('Normalized Alcohol (0 to 1)')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# 4. Z-Score Normalization on initial data for 'Alcohol' and 'Malic acid'
z_scaler = StandardScaler()
df_wine[['Alcohol_Zscore', 'Malic_acid_Zscore']] = z_scaler.fit_transform(
    df_wine[['Alcohol', 'Malic acid']]
)

# Plot Z-Score Normalized Malic Acid
plt.figure(figsize=(8, 5))
sns.histplot(df_wine['Malic_acid_Zscore'], kde=True, color='teal', bins=15)
plt.title('Distribution of Z-Score Normalized Malic Acid')
plt.xlabel('Z-Score Normalized Malic Acid')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()