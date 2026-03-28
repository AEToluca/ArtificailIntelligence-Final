from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


def load_features_and_target(csv_path: Path, target_col: str = "Performance"):
    """Load the dataset and split into features/target."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found. Found: {list(df.columns)}")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], save_path: Path):
    """Save a labeled confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Gradient Boosting Confusion Matrix",
    )

    # annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(model: GradientBoostingClassifier, feature_names: list[str], save_path: Path):
    """Save a horizontal bar chart of feature importances (top 20 by default)."""
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    if importances is None or len(importances) == 0:
        return

    # top 20 as it avoids overcrowding
    top_k = min(20, len(importances))
    idx = np.argsort(importances)[-top_k:][::-1]
    sorted_names = [feature_names[i] for i in idx]
    sorted_importances = importances[idx]

    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.25)))
    ax.barh(sorted_names[::-1], sorted_importances[::-1], color="#4a90e2")
    ax.set_xlabel("Relative Importance")
    ax.set_title("Gradient Boosting Feature Importance (Top 20)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    repo_root = Path(__file__).resolve().parent
    csv_path = repo_root / "cleaned_student_data.csv"

    X, y = load_features_and_target(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm_labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=cm_labels)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # Visuals,
    plot_confusion_matrix(cm, cm_labels, repo_root / "gb_confusion_matrix.png")
    plot_feature_importance(model, X.columns.tolist(), repo_root / "gb_feature_importance.png")
    print("\nSaved visualizations: gb_confusion_matrix.png, gb_feature_importance.png")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
