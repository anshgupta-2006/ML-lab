import numpy as np
import pandas as pd

# 1. Parse CSV replacing '?' with NaN and print header
df_att = pd.read_csv('attainment.csv', na_values='?')

print("--- Header ---")
print(df_att.columns.tolist())

# 2. Identify missing values
total_missing = df_att.isnull().sum().sum()
col_missing = df_att.isnull().sum()

print(f"\nTotal Missing Values in Dataset: {total_missing}")
print("\nMissing Values per Column:")
print(col_missing)

# Drop columns with > 50% missing values
threshold = 0.5 * len(df_att)
df_filtered = df_att.dropna(thresh=len(df_att) - threshold, axis=1)

# Save filtered DataFrame
df_filtered.to_csv('filtered.csv', index=False)
print("\nFiltered DataFrame saved to 'filtered.csv'. Dropped columns:", 
      set(df_att.columns) - set(df_filtered.columns))

# 3. Handle Missing Values based on Skewness
# Rule: Use median for heavily skewed data (|skew| > 1), otherwise use mean.
df_completed = df_filtered.copy()

for col in df_completed.select_dtypes(include=[np.number]).columns:
    if df_completed[col].isnull().sum() > 0:
        skew_val = df_completed[col].skew()
        
        if abs(skew_val) > 1.0:
            fill_val = df_completed[col].median()
            strategy = "Median (Heavily Skewed)"
        else:
            fill_val = df_completed[col].mean()
            strategy = "Mean (Slightly/Moderately Skewed)"
            
        df_completed[col] = df_completed[col].fillna(fill_val)
        print(f"Column '{col}' (Skewness: {skew_val:.2f}) -> Filled with {strategy}: {fill_val:.2f}")

print("\n--- Complete Data ---")
print(df_completed.head())