SVEC_Standard = 12.1

# 7Li/6Li sample = (d7Li/1000 + 1) * SVEC_Standard
df_copy['7Li/6Li'] = (df_copy['d7Li']/1000 + 1) * SVEC_Standard


import numpy as np
import matplotlib.pyplot as plt

# Define parameters
phi = 0.3  # Porosity

q = 1.0  # Water discharge (m/s)

k = 1.896e-4  # Reaction rate constant (1/s)

C0 = 0.16  # Inlet concentration (uM)

L = 100.0  # Length of the flow path (meters)

T = 1 * 86400  # Total travel time (seconds)

# Numerical parameters
Nx = 100  # Number of spatial points
Nt = 1000000 # Number of time steps
dx = L / Nx  # Spatial step size
dt = T / Nt  # Time step size

# Stability condition for explicit scheme
if dt > dx * phi / q:
    raise ValueError("Time step too large; explicit scheme may become unstable.")

# Initialize concentration array
C = np.zeros(Nx)  # Concentration at the current time step
C_new = np.zeros_like(C)  # Concentration at the next time step

# Initial condition: uniform concentration along the domain
C[:] = C0

# Boundary condition at x = 0: fixed concentration
C[0] = C0

# Time evolution
for n in range(Nt):
    for i in range(1, Nx):  # Skip the first point (boundary condition at x = 0)
        # Finite difference equation
        advective_term = (q / phi) * (C[i] - C[i - 1]) / dx
        reactive_term = k * C[i]
        C_new[i] = C[i] - dt * (advective_term + reactive_term)
    
    # Update concentration array
    C[:] = C_new[:]
    C[0] = C0  # Reapply boundary condition at x = 0

# Steady-state concentration
x = np.linspace(0, L, Nx)  # Spatial grid
travel_time = x * phi / q  # Convert distance to travel time

#convert travel time to days:
travel_time_days = travel_time / 86400


# Plot results as a function of travel time with log scale
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(travel_time_days, C, label="Concentration at steady state", lw=2)


#add data points for df_copy['Traverse'] == 'Traverse 5'
traverse_data = df_copy[df_copy['Traverse'] == 'Traverse 5']
#print(traverse_data['Li_uM'])

# Extract the range of Li_mM values from traverse_data
li_min = traverse_data['Li_uM'].min()
li_max = traverse_data['Li_uM'].max()

#print(f"Li_mM range for Traverse 5: {li_min:.2f} - {li_max:.2f} uM")

# Add the y-axis bar to represent the range
#ax.axhspan(li_min, li_max, color='lightblue', alpha=0.5, label="Li_uM range (Traverse 5)")


ax.set_xscale("log")  # Set x-axis to log scale
ax.set_title("Steady-State Concentration as a Function of Travel Time (Log Scale)")
ax.set_xlabel("Travel Time (t) [days] (log scale)")
ax.set_ylabel("Concentration (C) [uM]")
ax.grid(which="both", linestyle="--", linewidth=0.5)
ax.legend()

# Disable scientific notation and offset on the y-axis
ax.ticklabel_format(style='plain', axis='y', useOffset=False, useMathText=False)

plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from geopy.distance import geodesic


# Observed concentration data
traverse_data = df_copy[df_copy['Traverse'] == 'Traverse 5']

# Find the easternmost point
easternmost_point = traverse_data.loc[traverse_data["Longitude"].idxmax()]
start_point = (easternmost_point["Latitude"], easternmost_point["Longitude"])

# Calculate distances from the easternmost point
traverse_data.loc[:, "Position"] = traverse_data.apply(
    lambda row: geodesic((row["Latitude"], row["Longitude"]), start_point).meters, axis=1
)

# Define known parameters
phi = 0.3  # Porosity (dimensionless)
q = 1.0  # Water discharge (m/s)
k = 1.896e-4  # Reaction rate constant (1/s)
C0 = 0.16  # Inlet concentration (uM)
Nx = 5000  # Number of spatial points
Nt = 10000  # Number of time steps
dt = 1.0  # Time step size (adjustable for stability)

# Function to simulate the model for a given flow path length
def simulate_model_with_length(L):
    dx = L / Nx  # Spatial step size
    dt_stable = phi * dx / q  # Ensure stability condition
    dt_actual = min(dt, dt_stable)  # Use the smaller of given and stable time step
    x = np.linspace(0, L, Nx)  # Distance along the flow path
    C = np.zeros(Nx)  # Concentration array
    C_new = np.zeros_like(C)
    C[:] = C0
    C[0] = C0

    for n in range(Nt):
        for i in range(1, Nx):
            advective_term = (q / phi) * (C[i] - C[i - 1]) / dx
            reactive_term = k * C[i]
            C_new[i] = C[i] - dt_actual * (advective_term + reactive_term)
        C[:] = C_new[:]
        C[0] = C0

    travel_time = x * phi / q  # Calculate travel time for this flow path length
    return C, travel_time

# Objective function for optimization
def objective_function(L):
    C_simulated, _ = simulate_model_with_length(L)
    observed_positions = traverse_data["Position"]
    simulated_positions = np.linspace(0, L, Nx).flatten()  # Flatten ensures 1D
    
    print("Observed Positions Shape:", observed_positions.shape)
    print("Simulated Positions Shape:", simulated_positions.shape)
    print("Simulated Concentrations Shape:", C_simulated.shape)
    
    C_interpolated = np.interp(observed_positions, simulated_positions, C_simulated)
    mse = np.mean((C_interpolated - traverse_data["Li_uM"]) ** 2)  # Mean squared error
    return mse

# Initial guess for L
initial_guess_L = 10000.0  # Initial flow path length in meters

# Minimize the objective function
result = minimize(objective_function, initial_guess_L, bounds=[(1, 500)])  # Bounds for L
optimized_L = result.x[0]

# Simulate with optimized length
C_optimized, travel_time_optimized = simulate_model_with_length(optimized_L)

# Convert travel time to days for plotting
travel_time_days = travel_time_optimized / 86400


x_calculated = travel_time_optimized * q / phi

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(x_calculated, C_optimized, label="Optimized Model", lw=2)
plt.scatter(
    traverse_data["Position"], 
    traverse_data["Li_uM"], 
    color="red", 
    label="Observed Data",
)
plt.xscale("log")
plt.xlabel("Travel Time (days) (log scale)")
plt.ylabel("Concentration (uM)")
plt.title("Optimized Flow Path Length and Travel Time")
plt.grid(which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.show()

print(f"Optimized Flow Path Length: {optimized_L:.2f} meters")