"""Step 4: look at the data before modelling it."""
import pandas as pd

df = pd.read_csv("data/processed/adult_clean.csv")

print("=== 1. Size ===")
print(f"{df.shape[0]} rows, {df.shape[1]} columns\n")

print("=== 2. Column types ===")
print(df.dtypes, "\n")

print("=== 3. Missing values per column ===")
missing = df.isna().sum()
print(missing[missing > 0], "\n")

print("=== 4. Target balance (income) ===")
print(df["income"].value_counts(normalize=True).round(3), "\n")

print("=== 5. Numeric columns: ranges ===")
print(df.describe().round(1).T, "\n")

print("=== 6. Categorical columns: number of distinct values ===")
for col in df.select_dtypes(exclude="number").columns:
    print(f"{col:15} {df[col].nunique():3} values, e.g. {list(df[col].dropna().unique()[:3])}")

print("\n=== 7. Exact duplicate rows ===")
print(df.duplicated().sum())