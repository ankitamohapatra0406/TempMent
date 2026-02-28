import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

model = load_model("models/lstm_model.h5")

df = pd.read_csv("data/processed/ml_dataset.csv")

features = [
    "motion_score",
    "temperature",
    "humidity",
    "occupancy_duration",
    "time_since_last_motion",
    "peak_hour_frequency",
    "thermal_recovery_time",
    "average_cooling_time"
]

X = df[features]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Take last 10 timesteps
last_sequence = X_scaled[-10:]
last_sequence = np.expand_dims(last_sequence, axis=0)

prediction = model.predict(last_sequence)

probability = prediction[0][0]

print("Future Occupancy Probability:", probability)

if probability > 0.7:
    print("Recommendation: Pre-Cool Room")
else:
    print("Recommendation: Keep Current State")