import pandas as pd
#Inspection for the given data.csv to see if it needs data wrangling as it looks clean
# Load dataset
df = pd.read_csv("data.csv")

print(df.head())
print(df.info())
print(df.describe())

df = df.drop_duplicates()

df["verification.result"] = df["verification.result"].astype(int)

categorical_cols = ["property.product", "property.winner"]

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X_class = df.drop(["verification.result", "verification.time"], axis=1)
y_class = df["verification.result"]

X_reg = df.drop(["verification.time", "verification.result"], axis=1)
y_reg = df["verification.time"]

print("Data wrangling complete.")
