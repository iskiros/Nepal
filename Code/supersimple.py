import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load data
df2 = pd.read_excel('Datasets/Nepal Master Sheet.xlsx', sheet_name='Final_compiled')
df = df2.copy()



# Define Traverse assignment function (same as before)
def assign_traverse(gns):
    if not isinstance(gns, str):  # Handle non-string values
        return None
    gns = gns.split('22')[0].split('23')[0].strip("'").strip('"')
    if gns.startswith("S1"):
        return "Traverse 1*" if gns in ["S1m", "S1i"] else "Traverse 1"
    elif gns.startswith("S2"):
        return "Traverse 2"
    elif gns.startswith("S3"):
        if gns in ["S3k", "S3m", "S3u", "S3s", "S3ag", "S3ad"]:
            return "Traverse 4"
        elif gns in ["S3y", "S3ae"]:
            return "Traverse 3*"
        return "Traverse 3"
    elif gns.startswith("S4"):
        return "Traverse 5*" if gns in ["S4m", "S4l"] else "Traverse 5"
    return None



df["Traverse"] = df["GNS"].apply(assign_traverse)
df = df[df["Traverse"] == "Traverse 3"]



# Convert Li concentrations to millimolar
df['Li_mM'] = df['Li_ppm'] / 6.94

# Convert Li concentrations to nanomolar
df['Li_nM'] = df['Li_ppm'] * 1000000 / 6.94

# Convert Li concentrations to mol/m^3 (from nM)
df['Li_mol_m3'] = df['Li_nM'] * 1e-6  # Convert nM to mol/m^3


variable = "Li_mol_m3"





# Normalize z': Assign z'=1 to the maximum concentration
min_var = df[variable].min()
max_var = df[variable].max()


# Assume flow path length larger for a higher concentration
df["z'"] = (df[variable] - min_var) / (max_var - min_var)





# Reaction rate constant k (log10 form) and unit conversion
log_k = -11.2  # Example log10 k value
k = np.exp(log_k)  # Convert to mol/m^2/s
A_s = 1  # Assume unit specific surface area for simplicity


C_zero = df[variable].min()  # Initial concentration (mol/m^3)


# Function to calculate N_D
def calculate_ND(phi, t, C0, k, A_s):
    """
    Calculate the Damköhler number N_D.
    phi: porosity
    t: time (h/w, in seconds)
    C0: initial concentration (in mol/m^3)
    k: reaction rate constant (in mol/m^2/s)
    A_s: specific surface area (in m^2/m^3)
    """
    return t * k * A_s / (phi * C0)





def EE(z, N_D, f, C0):
    """
    Explicit Euler method to solve for C' as a function of z'.
    
    Parameters:
        z: array of nondimensional distances (z')
        N_D: Damköhler number
        f: reaction fraction
        C0: initial concentration (C' at z'=0)
    
    Returns:
        C: array of concentrations at each z'
    """

N_D = calculate_ND(0.07, 25 * 365 * 24 * 3600, C_zero, k, A_s)  # Damköhler number

frac = 0.5  # Fraction of reaction initial guess

f = lambda z, C: N_D * (1 - frac)  # ODE

h = 0.1 # Step size

z = np.arange(0, 1 + h, h) # Numerical grid

c0 = C_zero # Initial Condition


# Explicit Euler Method
c = np.zeros(len(z))
c[0] = c0

for i in range(0, len(z) - 1):
    c[i + 1] = c[i] + h*f(c[i], c[i])


plt.figure(figsize = (12, 8))
plt.plot(z, c, 'bo--', label='Approximate')
plt.plot(z,  (N_D * (1-frac)), 'g', label='Exact')
plt.title('Approximate and Exact Solution \
for Simple ODE')
plt.xlabel('z')
plt.ylabel('f(z)')
plt.grid()
plt.legend(loc='lower right')
plt.show()






# Initial guesses for porosity and time (h/w)
initial_guess = [0.07, 25 * 365 * 24 * 3600]  # 25 years in seconds
