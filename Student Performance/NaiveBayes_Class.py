import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# load clean data
data = pd.read_csv("Student Performance/cleaned_student_data.csv")

#imbalance check
print("Class counts:\n", data["Performance"].value_counts(), "\n")

# X is every column except the target
X = data.drop(columns=["Performance"])
# y is just the performance score
y = data["Performance"]

# 70/30 split for training and testing, keep class balance consistent
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Gaussian NB, continuous assumption
gauss_scaler = StandardScaler()
X_train_gauss = gauss_scaler.fit_transform(X_train)
X_test_gauss = gauss_scaler.transform(X_test)

gauss_model = GaussianNB()
gauss_model.fit(X_train_gauss, y_train)

gauss_preds = gauss_model.predict(X_test_gauss)

print("GaussianNB Accuracy:", accuracy_score(y_test, gauss_preds))
print("\nGaussianNB Confusion Matrix:\n", confusion_matrix(y_test, gauss_preds))
print("\nGaussianNB Classification Report:\n", classification_report(y_test, gauss_preds))

# Complement NB
# Was having issues with the large amount of group 4 data, Complement NB accounts for that and increased our accuracy
# keep inputs non-negative; MinMax preserves 0-1 range for counts/percentages
comp_scaler = MinMaxScaler()
X_train_comp = comp_scaler.fit_transform(X_train)
X_test_comp = comp_scaler.transform(X_test)

comp_model = ComplementNB(alpha=1.0)
comp_model.fit(X_train_comp, y_train)

comp_preds = comp_model.predict(X_test_comp)

print("ComplementNB Accuracy:", accuracy_score(y_test, comp_preds))
print("\nComplementNB Confusion Matrix:\n", confusion_matrix(y_test, comp_preds))
print("\nComplementNB Classification Report:\n", classification_report(y_test, comp_preds))
