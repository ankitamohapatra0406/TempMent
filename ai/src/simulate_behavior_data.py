import pandas as pd
import numpy as np

np.random.seed(42)

rows = []

for i in range(2000):
    motion = np.random.randint(0, 100)
    temperature = 22 + np.random.normal(0, 1)
    humidity = 50 + np.random.normal(0, 3)
    csi_variance = np.random.normal(0, 1)

    #Behavior logic simulation
    if motion < 10 and csi_variance < 0.5:
        label = 0  #Empty
    elif motion < 40:
        label = 1  #1-2 people
    elif motion < 75:
        label = 2  #5+ people
    elif csi_variance > 2:
        label = 3  #Door open
    else:
        label = 4  #Random disturbance

    rows.append([
        motion,
        temperature,
        humidity,
        csi_variance,
        label
    ])

df = pd.DataFrame(rows, columns=[
    "motion_score",
    "temperature",
    "humidity",
    "csi_variance",
    "behavior_label"
])

df.to_csv("data/processed/behavior_dataset.csv", index=False)

print("Behavior dataset generated.")