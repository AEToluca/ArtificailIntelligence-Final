import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# load clean data
data = pd.read_csv("cleaned_student_data.csv")

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


# --- VISUALIZATION: CONFUSION MATRICES ---
print("\n--- Generating Confusion Matrix Visuals ---")

# 1. Capture the confusion matrix data
cm_gauss = confusion_matrix(y_test, gauss_preds)
cm_comp = confusion_matrix(y_test, comp_preds)

# 2. Set up the figure with 2 subplots side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Impact of Algorithm Shift on Class Imbalance', fontsize=16, fontweight='bold', y=1.05)

# 3. Plot GaussianNB (The "Lazy" Model)
# Using a red/orange color map to imply it's the "bad/before" model
sns.heatmap(cm_gauss, annot=True, fmt='d', cmap='OrRd', ax=axes[0], cbar=False, annot_kws={"size": 14})
axes[0].set_title('GaussianNB\n(Majority Class Bias)', fontsize=14, pad=10)
axes[0].set_xlabel('Predicted Performance Group', fontsize=12)
axes[0].set_ylabel('Actual Performance Group', fontsize=12)
# Center labels for groups 1, 2, 3, 4
axes[0].set_xticks([0.5, 1.5, 2.5, 3.5])
axes[0].set_xticklabels(['1', '2', '3', '4'])
axes[0].set_yticks([0.5, 1.5, 2.5, 3.5])
axes[0].set_yticklabels(['1', '2', '3', '4'], rotation=0)

# 4. Plot ComplementNB (The "Fixed" Model)
# Using a green color map to imply it's the "good/after" model
sns.heatmap(cm_comp, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False, annot_kws={"size": 14})
axes[1].set_title('ComplementNB\n(Balanced Predictions)', fontsize=14, pad=10)
axes[1].set_xlabel('Predicted Performance Group', fontsize=12)
axes[1].set_ylabel('Actual Performance Group', fontsize=12)
axes[1].set_xticks([0.5, 1.5, 2.5, 3.5])
axes[1].set_xticklabels(['1', '2', '3', '4'])
axes[1].set_yticks([0.5, 1.5, 2.5, 3.5])
axes[1].set_yticklabels(['1', '2', '3', '4'], rotation=0)

# 5. Format and Save
plt.tight_layout()
output_filename = 'naive_bayes_comparison.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Success! Visual saved to your current folder as: {output_filename}")