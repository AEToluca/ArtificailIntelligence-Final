import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Load data
data = pd.read_csv("Auction/data.csv")

X = data.drop(columns=["verification.result", "verification.time"])
# regression target: predict verification.time
y = data["verification.time"]

# Pipeline = scaler + model
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression()),
])

# Cross-validation setup
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Out-of-fold predictions for regression metrics
preds = cross_val_predict(pipe, X, y, cv=cv, method="predict")

# Metrics
mae = mean_absolute_error(y, preds)
rmse = root_mean_squared_error(y, preds)
r2 = r2_score(y, preds)

print("MAE (CV):", mae)
print("RMSE (CV):", rmse)
print("R2 (CV):", r2)

# Plot actual vs predicted
plt.figure()
plt.scatter(y, preds, alpha=0.6)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Linear Regression: Actual vs Predicted (CV)")

min_val = min(y.min(), preds.min())
max_val = max(y.max(), preds.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray")
plt.show()

# Plot residuals
residuals = y - preds
plt.figure()
plt.hist(residuals, bins=30, alpha=0.7)
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Count")
plt.title("Linear Regression: Residuals (CV)")
plt.show()
