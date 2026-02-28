import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

rows = 1000
start_time = datetime.now()

data = []

for i in range(rows):
    timestamp = start_time + timedelta(minutes=i)

    motion = np.random.randint(0, 100)
    temperature = np.random.uniform(22, 30)
    humidity = np.random.uniform(40, 70)

    occupancy = 1 if motion > 30 else 0

    room_id = "Room_1"

    data.append([
        timestamp,
        motion,
        temperature,
        humidity,
        occupancy,
        room_id
    ])

df = pd.DataFrame(data, columns=[
    "timestamp",
    "motion_score",
    "temperature",
    "humidity",
    "occupancy",
    "room_id"
])

df.to_csv("data/raw/iot_raw_data.csv", index=False)

print("Raw data saved")