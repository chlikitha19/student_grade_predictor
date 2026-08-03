import pandas as pd

df = pd.read_csv("../data/student_data.csv")

print(df.head())

print("-" * 80)

df.info()

print("-" * 80)

print(df.shape)

print("-" * 80)

print(df.columns)

print("-" * 80)

print(df.describe())

print("-" * 80)

print(df.isnull().sum())

print("-" * 80)

print(df.duplicated().sum())

df = df.drop_duplicates()

df.to_csv("../data/clean_student_data.csv", index=False)

print("Cleaned dataset saved successfully!")