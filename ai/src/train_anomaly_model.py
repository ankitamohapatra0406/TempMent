import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

df = pd.read_csv("data/processed/freezer_normal.csv")

#FE
df["temp_gradient"] = df["temperature"].diff().fillna(0)

features = ["temperature", "temp_gradient"]
X = df[features]

model = IsolationForest(contamination=0.02)
model.fit(X)

joblib.dump(model, "models/freezer_anomaly.pkl")

print("Anomaly model trained and saved.")