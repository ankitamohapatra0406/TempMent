import pandas as pd
import joblib

model = joblib.load("models/freezer_anomaly.pkl")

df = pd.read_csv("data/processed/freezer_normal.csv")

# Inject artificial failure
df.loc[800:850, "temperature"] += 10  #simulating compressor failure

df["temp_gradient"] = df["temperature"].diff().fillna(0)

features = ["temperature", "temp_gradient"]
X = df[features]

preds = model.predict(X)
df["anomaly"] = preds

anomalies = df[df["anomaly"] == -1]

print("Anomalies detected at timestamps:")
print(anomalies[["time", "temperature"]].head())