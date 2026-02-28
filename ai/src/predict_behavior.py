import pandas as pd
import joblib

model = joblib.load("models/behavior_rf.pkl")

df = pd.read_csv("data/processed/behavior_dataset.csv")

features = [
    "motion_score",
    "temperature",
    "humidity",
    "csi_variance"
]

latest = df[features].iloc[-1:]

prediction = model.predict(latest)[0]

labels = {
    0: "Empty",
    1: "1-2 People",
    2: "5+ People",
    3: "Door Open",
    4: "Random Disturbance"
}

print("Detected Behavior:", labels[prediction])