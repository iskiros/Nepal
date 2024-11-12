# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # Load and preprocess your data
# data = pd.read_excel('Datasets/Nepal Master Sheet.xlsx')
# data = data[(data['Sample type'] == 'Spring water') & (data['Traverse'] == 'Traverse 1')]
# data['Sr_mM'] = data['Sr_ppm'] / 87.62  # Convert Sr from ppm to mM


# # Set up constants
# k = 1.896 * 10**(-4) #in seconds

# # Cmix = XaCo +  XbCoe^(-ktmix)

# # Co = initial concentration of Sr = highest concentration of Sr in the sample

# Co = data['Sr_mM'].max() # C max

# Cb = data['Sr_mM'].min() #C min

# # Xa = proportion between Co and Cmin

# data['Xa'] = (data['Sr_mM'] - Cb) / (Co - Cb)

# data['Xb'] = 1 - data['Xa']

# # Cmix = XaCo +  XbCoe^(-ktmix)

# # get t out of it

# #data['tmix'] = -(1/k) * np.log((data['Sr_mM'] - data['Xa']*Co) / (data['Xb']*Co)) 

# data['tmix'] = -(1/k) * np.log((data['Sr_mM'] - data['Xa']*Co) / (data['Xb']*Co))



# #print((data['Sr_mM'] - data['Xa']*Co) / (data['Xb']*Co))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess your data
data = pd.read_excel('Datasets/Nepal Master Sheet.xlsx')
data = data[(data['Sample type'] == 'Spring water') & (data['Traverse'] == 'Traverse 1')]
data['Sr_mM'] = data['Sr_ppm'] / 87.62  # Convert Sr from ppm to mM

# Set up constants
k = 1.896 * 10**(-4)  # in seconds

# Define Co based on the highest Sr concentration (initial concentration)
Co = data['Sr_mM'].max()  # C max

# Calculate tmix using the simpler Rayleigh fractionation formula
tmix_values = []

for index, row in data.iterrows():
    Sr_mM = row['Sr_mM']
    
    # Calculate tmix directly
    if Sr_mM > 0:  # Ensure we don't take log of zero or a negative
        tmix = -(1 / k) * np.log(Sr_mM / Co)
    else:
        tmix = np.nan  # Set to NaN if Sr_mM is non-positive
    
    # Append tmix value for inspection
    tmix_values.append(tmix)
    
    # Print intermediate values for this sample
    #print(f"Sample index {index}:")
    #print(f"  Sr_mM = {Sr_mM}")
    #print(f"  tmix = {tmix}")
    #print("-" * 30)

# Add results to the DataFrame for analysis
data['tmix'] = tmix_values

# Display the results DataFrame
#print(data[['Sr_mM', 'tmix']])


# Print concentration against time
# Convert tmix from seconds to days for plotting
data['tmix_hours'] = data['tmix'] / (3600)
data['tmix_days'] = data['tmix_hours'] / 24

# Plot concentration against time in days
plt.figure(figsize=(10, 6))
plt.plot(data['tmix_days'], data['Sr_mM'], 'o', markersize=5)
plt.xlabel('Time (days)')
plt.xscale('log')   
plt.ylabel('Sr Concentration (mM)')
plt.title('Sr Concentration vs. Time')
plt.grid(True)
plt.show()
