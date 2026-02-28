import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/clean_data.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

#Occupancy Duration
df["occupancy_change"] = df["occupancy"].diff().fillna(0)
df["occupancy_group"] = (df["occupancy_change"] != 0).cumsum()

occupancy_duration = df.groupby("occupancy_group")["timestamp"].transform(
    lambda x: (x.max() - x.min()).total_seconds() / 60
)

df["occupancy_duration"] = occupancy_duration

#Time Since Last Motion
df["last_motion_time"] = df["timestamp"].where(df["motion_score"] > 30)
df["last_motion_time"] = df["last_motion_time"].ffill()

df["time_since_last_motion"] = (
    df["timestamp"] - df["last_motion_time"]
).dt.total_seconds() / 60

#Peak Hour Frequency
df["hour"] = df["timestamp"].dt.hour
hour_counts = df.groupby("hour")["occupancy"].mean()
df["peak_hour_frequency"] = df["hour"].map(hour_counts)

#Thermal Recovery Time
df["thermal_recovery_time"] = 0

for i in range(1, len(df)):
    if df.loc[i-1, "occupancy"] == 1 and df.loc[i, "occupancy"] == 0:
        base_temp = df.loc[i, "temperature"]
        for j in range(i, len(df)):
            if df.loc[j, "temperature"] <= base_temp - 2:
                delta = (
                    df.loc[j, "timestamp"] - df.loc[i, "timestamp"]
                ).total_seconds() / 60
                df.loc[i, "thermal_recovery_time"] = delta
                break

#Average Cooling Time
df["temp_diff"] = df["temperature"].diff()
df["cooling_time"] = df["temp_diff"].apply(lambda x: 1 if x < 0 else 0)

df["average_cooling_time"] = df["cooling_time"].rolling(window=10).mean()
df["average_cooling_time"] = df["average_cooling_time"].fillna(0)

#FUTURE TARGET (10 minutes ahead)If 1 row = 1 minute
df["future_occupancy"] = df["occupancy"].shift(-10)

df = df.dropna()

df.to_csv("data/processed/ml_dataset.csv", index=False)

print("Feature engineering complete.")