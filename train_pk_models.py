import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# 1. Load dataset
df = pd.read_csv("Crop_recommendation_9_features.csv")

# 2. Features available from your APIs
features = [
    "nitrogen",
    "ph",
    "clay",
    "organic_carbon",
    "temperature",
    "humidity",
    "rainfall"
]

X = df[features]


# =========================================================
# PREDICT PHOSPHORUS
# =========================================================

y_p = df["P"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_p,
    test_size=0.20,
    random_state=42
)

p_model = RandomForestRegressor(
    n_estimators=400,
    random_state=42,
    n_jobs=-1
)

p_model.fit(X_train, y_train)

p_prediction = p_model.predict(X_test)

p_r2 = r2_score(y_test, p_prediction)
p_mae = mean_absolute_error(y_test, p_prediction)
p_rmse = mean_squared_error(y_test, p_prediction) ** 0.5

print("\n========== PREDICTION MODEL - P ==========")
print("R2 Score :", p_r2)
print("MAE      :", p_mae)
print("RMSE     :", p_rmse)

joblib.dump(p_model, "p_model.pkl")

print("P model saved as p_model.pkl")


# =========================================================
# PREDICT POTASSIUM
# =========================================================

y_k = df["K"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_k,
    test_size=0.20,
    random_state=42
)

k_model = RandomForestRegressor(
    n_estimators=400,
    random_state=42,
    n_jobs=-1
)

k_model.fit(X_train, y_train)

k_prediction = k_model.predict(X_test)

k_r2 = r2_score(y_test, k_prediction)
k_mae = mean_absolute_error(y_test, k_prediction)
k_rmse = mean_squared_error(y_test, k_prediction) ** 0.5

print("\n========== PREDICTION MODEL - K ==========")
print("R2 Score :", k_r2)
print("MAE      :", k_mae)
print("RMSE     :", k_rmse)

joblib.dump(k_model, "k_model.pkl")

print("K model saved as k_model.pkl")