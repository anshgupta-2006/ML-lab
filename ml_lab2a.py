import numpy as np
import pandas as pd

# 1. Read Employee_list dataset with Name as index
# Adjust the file path/extension as per your dataset (e.g., 'Employee_list.csv' or 'Employee_list.xlsx')
df_emp = pd.read_csv('Employee_list.csv', index_col='Name')

# Extract required columns
df_emp = df_emp[['Age', 'Profession', 'Salary']]
print("--- Employee Data ---")
print(df_emp.head())

# 2. Calculate Median, Mode, Skewness, and Kurtosis for Salary
salary_median = df_emp['Salary'].median()
salary_mode = df_emp['Salary'].mode()[0]
salary_skew = df_emp['Salary'].skew()
salary_kurt = df_emp['Salary'].kurtosis()

print("\n--- Salary Statistics ---")
print(f"Median  : {salary_median}")
print(f"Mode    : {salary_mode}")
print(f"Skewness: {salary_skew:.4f}")
print(f"Kurtosis: {salary_kurt:.4f}")

# 3. Categorical Encoding for Profession (Engineer: 0, Doctor: 1, Teacher: 2)
profession_mapping = {'Engineer': 0, 'Doctor': 1, 'Teacher': 2}

# Use map to replace values; handles unlisted professions gracefully if needed
df_emp['Profession_Code'] = df_emp['Profession'].map(profession_mapping)

print("\n--- Encoded Employee Data ---")
print(df_emp.head())