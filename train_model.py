"""
Electricity Consumption Predictor
Foundation Project: Python + Pandas + Visualization + Machine Learning

Run:
    python train_model.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

DATA_FILE = "electricity_consumption.csv"

df = pd.read_csv(DATA_FILE)
df["datetime"] = pd.to_datetime(df["datetime"])

features = [
    "hour",
    "day_of_week",
    "is_weekend",
    "temperature_C",
    "humidity_percent",
    "previous_consumption_kWh"
]
target = "consumption_kWh"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = RandomForestRegressor(
    n_estimators=150,
    random_state=42,
    max_depth=12
)
model.fit(X_train, y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)

print("\n===== ELECTRICITY CONSUMPTION PREDICTOR =====")
print(f"MAE  : {mae:.3f} kWh")
print(f"RMSE : {rmse:.3f} kWh")
print(f"R²   : {r2:.3f}")

# Visualization 1: actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label="Actual")
plt.plot(pred[:100], label="Predicted")
plt.title("Actual vs Predicted Electricity Consumption")
plt.xlabel("Test Sample")
plt.ylabel("Consumption (kWh)")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()

# Visualization 2: feature importance
importance = pd.Series(model.feature_importances_, index=features).sort_values()
plt.figure(figsize=(8, 5))
importance.plot(kind="barh")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

joblib.dump(model, "electricity_model.pkl")
print("\nModel saved as electricity_model.pkl")
