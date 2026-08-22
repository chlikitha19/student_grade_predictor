import pandas as pd

# Load raw dataset
df = pd.read_csv("data/student_performance.csv")

# Inspect
print(df.head())
print(df.shape)
print(df.columns)
df.info()

# Check missing values
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Save cleaned dataset
df.to_csv(
    "data/student_performance_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")