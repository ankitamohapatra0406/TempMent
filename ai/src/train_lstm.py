import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load dataset
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
y = df["future_occupancy"]

# Scale
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Convert to sequences
def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y.iloc[i+time_steps])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42
)

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=20, batch_size=32)

model.save("models/lstm_model.h5")

print("Model trained and saved.")