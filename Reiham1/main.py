import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os

print("Current working directory:")
print(os.getcwd())

print("\nFiles in current directory:")
print(os.listdir())

print("Loading CSV files...")

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train = pd.read_csv(os.path.join(BASE_DIR, "mnist_train.csv"))
test = pd.read_csv(os.path.join(BASE_DIR, "mnist_test.csv"))

# Remove rows with missing values
train = train.dropna()
test = test.dropna()

X_train = train.iloc[:, 1:].values / 255.0
y_train = train.iloc[:, 0].values.astype(int)

X_test = test.iloc[:, 1:].values / 255.0
y_test = test.iloc[:, 0].values.astype(int)

print("Training data:", X_train.shape)
print("Training labels:", y_train.shape)
print("Testing data:", X_test.shape)
print("Testing labels:", y_test.shape)

rf_model = RandomForestClassifier(
    random_state=42,
    n_estimators=100
)

print("Training model...")
rf_model.fit(X_train, y_train)

y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

precision = precision_score(
    y_test,
    y_test_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_test_pred,
    average="weighted",
    zero_division=0
)

print("\n========== RESULTS ==========")
print("Training Accuracy :", train_acc)
print("Testing Accuracy  :", test_acc)
print("Precision         :", precision)
print("Recall            :", recall)

sample = X_test[0].reshape(1, -1)
prediction = rf_model.predict(sample)

print("\nActual Label    :", y_test[0])
print("Predicted Label :", prediction[0])

plt.imshow(X_test[0].reshape(28, 28), cmap="gray")
plt.title(f"Actual: {y_test[0]} | Predicted: {prediction[0]}")
plt.axis("off")
plt.show()
