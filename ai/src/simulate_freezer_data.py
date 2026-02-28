import pandas as pd
import numpy as np

np.random.seed(42)

rows = []
temp = -20  # Freezer baseline

for i in range(1500):

    # Simulate door open
    if np.random.rand() < 0.02:
        temp += np.random.uniform(3, 8)

    # Normal compressor recovery
    elif temp > -20:
        temp -= np.random.uniform(0.2, 0.6)

    # Random noise
    temp += np.random.normal(0, 0.2)
    rows.append([i, temp])

df = pd.DataFrame(rows, columns=["time", "temperature"])
df.to_csv("data/processed/freezer_normal.csv", index=False)

print("Freezer normal dataset created.")