import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ==========================
# قراءة البيانات
# ==========================
df = pd.read_csv(r"C:\Users\shima\OneDrive\Documents\GitHub\summer-intern-2026-02\shimaa\mnist project\wheat prediction\yield_df.csv")

# حذف العمود غير المهم
df = df.drop(columns=["Unnamed: 0"])

# حذف أي قيم ناقصة
df = df.dropna()

# ==========================
# تحديد الـ Features والـ Target
# ==========================
df = df.drop(columns=["pesticides_tonnes"])
X = df.drop("hg/ha_yield", axis=1)
y = df["hg/ha_yield"]

# الأعمدة النصية
categorical_features = ["Area", "Item"]

# الأعمدة الرقمية
numeric_features = [
    "Year",
    "average_rain_fall_mm_per_year",
    "avg_temp"
]


# تحويل الأعمدة النصية إلى أرقام
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# إنشاء Pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# تدريب الموديل
model.fit(X_train, y_train)

# التنبؤ
pred = model.predict(X_test)

# تقييم الموديل
print("R2 Score :", r2_score(y_test, pred))
print("MAE :", mean_absolute_error(y_test, pred))

# حفظ الموديل
joblib.dump(model, "model.pkl")

print("\nModel saved successfully as model.pkl")
