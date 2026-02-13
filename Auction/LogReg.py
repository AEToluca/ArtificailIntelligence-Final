import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("Auction/data.csv")

X = data.drop(columns=["verification.result", "verification.time"])
y = data["verification.result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=5000)
model.fit(X_train_scaled, y_train)

preds = model.predict(X_test_scaled)

acc = accuracy_score(y_test, preds)
print("Accuracy:", acc)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))

print("\nClassification Report:\n", classification_report(y_test, preds))
