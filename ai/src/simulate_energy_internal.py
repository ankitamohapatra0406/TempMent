import pandas as pd
import numpy as np

np.random.seed(42)

rows = []

for i in range(3000):
    current_temp = np.random.uniform(22, 35)
    occupancy_prob = np.random.uniform(0, 1)
    room_size = np.random.uniform(20, 100)
    thermal_recovery_speed = np.random.uniform(0.2, 0.8)
    temp_gradient = np.random.uniform(-0.5, 1.5)

    # Realistic runtime behavior
    runtime = (
        (current_temp - 22) * 2
        + occupancy_prob * 12
        + room_size * 0.04
        - thermal_recovery_speed * 5
        + temp_gradient * 3
    )

    runtime += np.random.normal(0, 1)

    rows.append([
        current_temp,
        occupancy_prob,
        room_size,
        thermal_recovery_speed,
        temp_gradient,
        runtime
    ])

df = pd.DataFrame(rows, columns=[
    "current_temp",
    "occupancy_prob",
    "room_size",
    "thermal_recovery_speed",
    "temp_gradient",
    "optimal_runtime"
])

df.to_csv("data/processed/internal_energy_dataset.csv", index=False)

print("Internal energy dataset created.")