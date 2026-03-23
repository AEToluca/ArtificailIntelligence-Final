import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
# Change this:
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# To this:
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score


# load the dataset
data = pd.read_csv("Auction/data.csv")

# separate features and target
X = data.drop(columns=["verification.result", "verification.time"])
# regression target: predict verification.time
y = data["verification.time"]

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# build a pipeline so scaling + knn happen together safely
pipe = Pipeline([
    ("scaler", StandardScaler()),      # scale the features
    ("knn", KNeighborsRegressor())     # run knn
])

# tell gridsearch what hyperparameters to try
param_grid = {
    "knn__n_neighbors": list(range(1, 51)),   # try k from 1 to 50
    "knn__weights": ["uniform", "distance"],  # equal vote vs distance-weighted vote
    "knn__p": [1, 2]                          # manhattan vs euclidean distance
}

# run cross validation and score using RMSE (lower is better)
grid = GridSearchCV(
    pipe,
    param_grid,
    scoring="neg_root_mean_squared_error",
    cv=5
)

# fit only on training data
grid.fit(X_train, y_train)

# print best hyperparameters found
print("best parameters found:", grid.best_params_)
print("best cross-validated RMSE:", -grid.best_score_)

# grab the best model
best_model = grid.best_estimator_

# make predictions on test set
preds = best_model.predict(X_test)

# print performance report
mae = mean_absolute_error(y_test, preds)
rmse = root_mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("\nregression metrics:\n")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")


# grab full cross-validation results
results = grid.cv_results_

# pull out hyperparameter values that were tested
k_vals = np.array([p["knn__n_neighbors"] for p in results["params"]])
w_vals = np.array([p["knn__weights"] for p in results["params"]])
p_vals = np.array([p["knn__p"] for p in results["params"]])

# get average RMSE and variation across folds
mean_rmse = -results["mean_test_score"]
std_rmse  = results["std_test_score"]

# identify the best performing combo
best_idx = grid.best_index_
best_k   = results["params"][best_idx]["knn__n_neighbors"]
best_rmse  = mean_rmse[best_idx]

# plot how RMSE changes across k values
plt.figure(figsize=(10, 6))
plt.title("knn hyperparameter search (5-fold cv)")
plt.xlabel("number of neighbors (k)")
plt.ylabel("mean cross-validated RMSE")

combos = [
    ("uniform", 1),
    ("uniform", 2),
    ("distance", 1),
    ("distance", 2),
]

# draw a line for each weights + distance combo
for w, p in combos:
    mask = (w_vals == w) & (p_vals == p)

    ks = k_vals[mask]
    ms = mean_rmse[mask]
    ss = std_rmse[mask]

    # sort by k so lines look clean
    order = np.argsort(ks)
    ks, ms, ss = ks[order], ms[order], ss[order]

    label = f"weights={w}, p={p}"
    plt.plot(ks, ms, label=label)

    # shaded region shows how much scores varied across folds
    plt.fill_between(ks, ms - ss, ms + ss, alpha=0.15)

# mark the best k
plt.scatter(best_k, best_rmse, s=100)
plt.axvline(best_k, linestyle="--")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

residuals = y_test - preds
plt.figure(figsize=(8, 5))
plt.scatter(preds, residuals, alpha=0.5, color='teal')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Verification Time')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('KNN Residual Plot (k=3)')
plt.show()