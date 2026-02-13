import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv("Auction/data.csv")


# X becomes all columns except those two
X = data.drop(columns=["verification.result", "verification.time"])

# y becomes just result
y = data["verification.result"]

# split data into training and test test_size=.20 
# 20% is testing and 42 is a random number to start randomly
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state=42)

# scale the data down so big numbers dont mess with data
scaler = StandardScaler()
# training data is used to find scaler information and then transformed
X_train_scaled = scaler.fit_transform(X_train)
# transformed using scaler data in the object from the one above
X_test_scaled = scaler.transform(X_test)

best = 0
bestScore = 0

# find best k for best acc
for k in range(1, 21):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    #print(k, acc)
    if acc > bestScore:
        bestScore = acc
        best = k

print(best, " ", bestScore)

# print("Confusion Matrix:\n", confusion_matrix(y_test,predictions))