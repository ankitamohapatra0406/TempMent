import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib

df = pd.read_csv("data/processed/internal_energy_dataset.csv")

features = [
    "current_temp",
    "occupancy_prob",
    "room_size",
    "thermal_recovery_speed",
    "temp_gradient"
]

X = df[features]
y = df["optimal_runtime"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor()
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, preds))

joblib.dump(model, "models/internal_energy_optimizer.pkl")

print("Energy optimizer saved.")