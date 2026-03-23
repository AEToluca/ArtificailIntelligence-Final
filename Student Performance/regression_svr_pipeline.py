import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


def build_preprocessor(df: pd.DataFrame, target_col: str) -> ColumnTransformer:
    """Create a ColumnTransformer that scales numerics and one-hot-encodes categoricals."""
    feature_df = df.drop(columns=[target_col])

    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def main() -> None:
    TARGET_COLUMN = "Performance"

    df = pd.read_csv("cleaned_student_data.csv")

    bool_cols = df.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"TARGET_COLUMN '{TARGET_COLUMN}' not found in CSV columns: {list(df.columns)}"
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Build preprocessing transformer.
    preprocessor = build_preprocessor(df, TARGET_COLUMN)

    # Combine preprocessing with the SVR model.
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")),
        ]
    )

    # Train/test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Fit model and predict.
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate.
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Report metrics.
    print("\nModel Evaluation (SVR)")
    print("----------------------")
    print(f"MAE: {mae:,.4f}")
    print(f"MSE: {mse:,.4f}")
    print(f"R2 : {r2:,.4f}")


if __name__ == "__main__":
    main()
