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


# --- OUTLIER DETECTION (IQR METHOD) ---
print("\n--- Checking for Outliers ---")
# Select only continuous numerical columns (ignore target/IDs if necessary)
# For the auction data, verification.time is a continuous variable
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds for what is considered an outlier
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Count how many rows fall outside these bounds
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    if len(outliers) > 0:
        print(f"Found {len(outliers)} potential outliers in '{col}'")
    else:
        print(f"No outliers found in '{col}'")
print("-----------------------------\n")

print("Data wrangling complete.")
