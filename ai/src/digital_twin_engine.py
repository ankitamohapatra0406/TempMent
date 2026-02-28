import joblib
import numpy as np

model = joblib.load("models/internal_energy_optimizer.pkl")

#Baseline real building state
current_temp = 30
occupancy_prob = 0.8
room_size = 60
thermal_recovery_speed = 0.5
temp_gradient = 0.3

baseline_runtime = model.predict([[
    current_temp,
    occupancy_prob,
    room_size,
    thermal_recovery_speed,
    temp_gradient
]])[0]

#Scenario:occupancy reduces by 40%
new_occupancy = occupancy_prob * 0.6

new_runtime = model.predict([[
    current_temp,
    new_occupancy,
    room_size,
    thermal_recovery_speed,
    temp_gradient
]])[0]

runtime_saved = baseline_runtime - new_runtime

#Assume compressor power = 3 kW
energy_saved_kwh = runtime_saved * 3 / 60

#Cost per kWh = ₹8
cost_saved = energy_saved_kwh * 8

#Carbon factor = 0.82 kg CO₂ per kWh
carbon_saved = energy_saved_kwh * 0.82

print("Baseline Runtime:", round(baseline_runtime, 2), "minutes")
print("New Runtime:", round(new_runtime, 2), "minutes")
print("Energy Saved:", round(energy_saved_kwh, 2), "kWh")
print("Cost Saved:", round(cost_saved, 2), "₹")
print("Carbon Reduced:", round(carbon_saved, 2), "kg CO₂")