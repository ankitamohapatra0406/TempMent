import pandas as pd

df = pd.read_csv("data/raw/iot_raw_data.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp")

df.to_csv("data/processed/clean_data.csv", index=False)

print("Preprocessing complete.")