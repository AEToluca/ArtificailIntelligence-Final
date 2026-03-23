import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# load the cleaned dataset 
data = pd.read_csv("cleaned_student_data.csv")

# the target column is 'Performance'
X = data.drop(columns=["Performance"])
y = data["Performance"]

# split the data (80% train / 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# scale features so that KNN distance isn't dominated by large values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# search for the best k in a small range
best_k = None
best_score = 0.0
for k in range(1, 21):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    if acc > best_score:
        best_score = acc
        best_k = k

print(f"best k = {best_k} with accuracy = {best_score:.4f}")

# train final model with best k and show evaluation metrics
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train_scaled, y_train)
preds = final_model.predict(X_test_scaled)

print("\nClassification report:\n", classification_report(y_test, preds))
print("Confusion matrix:\n", confusion_matrix(y_test, preds))
