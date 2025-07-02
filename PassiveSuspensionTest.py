import numpy as np
import matplotlib.pyplot as plt
from Suspension_Model import QuarterCarModel
from Road_profile import SquareWaveProfile

# --- Simulation Setup ---
sim_duration = 6.0  # seconds
dt = 0.005          # time step

# Instantiate the model and the road profile generator
model = QuarterCarModel(dt=dt)
road = SquareWaveProfile(period=3.0, amplitude=0.02) # 3-second period, 2cm bump

# --- Simulation Loop ---
num_steps = int(sim_duration / dt)
time_vector = np.linspace(0, sim_duration, num_steps)

# Data logging
history = {
    'time': time_vector,
    'x_s': np.zeros(num_steps),
    'x_u': np.zeros(num_steps),
    'x_g': np.zeros(num_steps),
    'x_s_ddot': np.zeros(num_steps)
}

# Get the full road profile in one go for efficiency
history['x_g'] = road.get_profile(time_vector)

# Run the simulation for the passive system (u=0)
for i in range(num_steps):
    # For a passive system, the control force is always zero
    control_force = 0.0
    
    # Get the current road input
    current_x_g = history['x_g'][i]
    
    # Step the model
    state, x_s_ddot, _ = model.step(control_force, current_x_g)
    
    # Log data
    history['x_s'][i] = state[0]
    history['x_u'][i] = state[2]
    history['x_s_ddot'][i] = x_s_ddot

# --- Plotting Results ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot Displacements
axs[0].plot(history['time'], history['x_s'] , label='Vehicle Body ($x_s$)')
axs[0].plot(history['time'], history['x_u'] , label='Wheel ($x_u$)', linestyle='--')
axs[0].plot(history['time'], history['x_g'] , label='Road Profile ($x_g$)', linestyle=':', color='gray')
axs[0].set_ylabel('Displacement (m)')
axs[0].legend()
axs[0].set_title('Passive Suspension Response to Square Wave')
axs[0].grid(True)

# Plot Body Acceleration
axs[1].plot(history['time'], history['x_s_ddot'], label='Body Acceleration ($\ddot{x}_s$)')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Acceleration ($m/s^2$)')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
