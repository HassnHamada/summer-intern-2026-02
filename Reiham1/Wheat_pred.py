import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==========================
# Load Dataset
# ==========================
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "mock_india_data_200k.csv")

df = pd.read_csv(csv_path)
print(df.corr(numeric_only=True)["Yield"].sort_values(ascending=False))

# Check for missing values
print(df.isnull().sum())
print(df.info())
print(df.isnull().sum())
print(df["Yield"].describe())

# Use only 20,000 samples for faster training
df = df.sample(n=20000, random_state=42)
# ==========================

# Uncomment this for faster testing

print("Dataset loaded successfully!")

# ==========================
# Features and Target
# ==========================
print("Preparing features...")

X = df.drop("Yield", axis=1)
y = df["Yield"]

# ==========================
# Split Data
# ==========================
print("Splitting data...")
# ==========================
# Split Data
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Create Random Forest Model
# ==========================
model = RandomForestRegressor(
    n_estimators=20,   # Use fewer trees for testing
    random_state=42,
    n_jobs=-1          # Use all CPU cores
)
print("Training model...")
model.fit(X_train, y_train)
print("Training complete!")
# ==========================
# Train Model
# ==========================
model.fit(X_train, y_train)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

print(importance.sort_values(by="Importance", ascending=False))

print("Making predictions...")
y_pred = model.predict(X_test)
# ==========================
# Prediction
# ==========================
y_pred = model.predict(X_test) 
# ==========================
# Evaluation
# ==========================
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R² Score:", r2)

# ==========================
# Compare Actual vs Predicted
# ==========================
results = pd.DataFrame({
    "Actual Yield": y_test.values,
    "Predicted Yield": y_pred
})

print(results.head(10))
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Yield")


plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red')

plt.show()