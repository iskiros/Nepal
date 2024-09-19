import matplotlib.pyplot as plt
import sys
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import os
import pandas as pd
import glob

def plot_gran_titration(SampleID, df, best_subset, intercept, coefficient, A, best_r2, vx):
    gran_marker_color = 'black'
    pH_marker_color = 'black'

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 7))

    # Gran vs Volume of Acid Added
    sns.lineplot(x='vol (L)', y='gran', data=df, ax=ax1, color='blue', label='Gran Value')
    sns.lineplot(x='vol (L)', y='gran', data=best_subset, ax=ax1, color='red')

    sns.scatterplot(x='vol (L)', y='gran', data=df, ax=ax1, color=gran_marker_color, 
                    marker='x', s=60, alpha=1, linewidth=1.5)

    # Calculate the x-value where the regression line hits y=0
    x_zero = -intercept / coefficient

    # Determine the x-values for the line extension: starting from the last point of the best subset to x_zero
    x_extension = np.linspace(best_subset['vol (L)'].iloc[-1], x_zero)

    # Calculate the y-values for these x-values using the regression equation
    y_extension = intercept + coefficient * x_extension

    # Plot the extension
    ax1.plot(x_extension, y_extension, '--', color='red')

    # Add vertical grid lines
    ax1.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)

    ax1.set_xlabel('Volume of Acid Added (L)')
    ax1.set_ylabel('Gran Value', color='blue', position=(0, 0.5))
    ax1.tick_params('y', colors='blue')
    legend = ax1.legend(loc='upper left')
    legend.set_bbox_to_anchor((0, 0.85))  # Adjust the second value to move the legend up or down

    ax1.set_title(f'Gran Titration - {SampleID}', fontsize=11)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    sns.lineplot(x='vol (L)', y='pH', data=df, ax=ax2, color='green', label='pH')
    sns.scatterplot(x='vol (L)', y='pH', data=df, ax=ax2, color=pH_marker_color, marker='+', s=60, alpha=1, linewidth=1.5)

    ax2.set_ylabel('pH', color='green')
    ax2.tick_params('y', colors='green')

    # Adjust the position of the pH legend
    legend_ph = ax2.legend(loc='upper left')
    legend_ph.set_bbox_to_anchor((0, 0.77))

    # Display the regression equation and alkalinity on the graph
    equation = f"Alkalinity = {A:.2f} \u03BCmol \nR² = {best_r2:.2f} \nVx = {(vx/1000):.2f} ml"
    props = dict(boxstyle='round,pad=0.3', facecolor='yellow', edgecolor='black')
    ax1.text(0.015, 0.67, equation, transform=ax1.transAxes, verticalalignment='top', bbox=props)

    fig.tight_layout()  # to make sure the right y-label is not slightly clipped
    
    # Save plot with dynamic filename
    plot_filename = f"/Users/enrico/PeruRiversProject/nepal/Plots/Gran_plot_{SampleID}.png"
    plt.savefig(plot_filename, dpi=300)
    plt.show()

# Search for csv files in the raw data folder
files = glob.glob("data/raw/*.csv")

output = pd.DataFrame()

# My version:
# for file in files:
#     df = pd.read_csv(file)
    
#     # Get the SampleID from the filename
#     SampleID = os.path.basename(file).split(".")[0]
    
#     # Convert volume to L
#     digit = 1.2
#     df['vol (L)'] = df['vol (digit)'] * 1e-3 * digit

#     # Compute the Gran function
#     def gran(vol, pH, v0=50):
#         return (v0 + vol) * 10 ** -pH

#     # Special case for specific SampleID
#     v0 = 30 if SampleID == "NEP-24-022" else 50
        
#     df['gran'] = gran(df['vol (L)'], df['pH'], v0)

#     # Filter pH values within the desired range
#     filtered_values = df[(df['pH'] <= 4.5) & (df['pH'] >= 3.35)]

#     def find_best_subset(filtered_values):
#         best_r2 = 0
#         best_subset = None

#         # Loop over all possible subsets of at least 4 points
#         for i in range(len(filtered_values) - 4):
#             for j in range(i + 4, len(filtered_values)):
#                 subset = filtered_values.iloc[i:j+1]
#                 X = sm.add_constant(subset['vol (L)'])  # Adds a constant term to the predictor
#                 y = subset['gran']
#                 model = sm.OLS(y, X)
#                 results = model.fit()
                
#                 r2 = results.rsquared
                
#                 if r2 > best_r2:
#                     best_r2 = r2
#                     best_subset = subset

#         return best_subset, best_r2

#     best_subset, best_r2 = find_best_subset(filtered_values)
#     X_best = sm.add_constant(best_subset['vol (L)'])  # Adds a constant term to the predictor
#     model_best = sm.OLS(best_subset['gran'], X_best)
#     results_best = model_best.fit()

#     intercept = results_best.params[0]
#     coefficient = results_best.params[1]

#     # Calculate the x-intercept in µL
#     vx = -intercept / coefficient * 1000

#     # Calculate the alkalinity in µmol/L
#     N = 0.05
#     A = vx * N / v0 * 10**3
    
#     print(SampleID + ' has an alkalinity of ' + str(A))

#     # Add alkalinity, R2, and x-intercept to the output dataframe
#     output_i = {'SampleID': SampleID, 'Alkalinity (uM)': A, 'R2': best_r2, 'vx (µL)': vx}
    
#     # Convert to dataframe and concatenate with output dataframe
#     output = pd.concat([output, pd.DataFrame(output_i, index=[0])], ignore_index=True)



# Al's version
for file in files:
    df = pd.read_csv(file)
    
    # get the SampleID from the filename
    SampleID = os.path.basename(file).split(".")[0]
    
    # Convert volume to L
    digit = 1.2
    df['vol (L)'] = df['vol (digit)'] * 1e-3 * digit

    # Compute the Gran function
    def gran(vol, pH, v0=50):
        return (v0 + vol) * 10 ** -pH

    if SampleID == "NEP-24-022":
        v0 = 30
    else:
        v0 = 50
        
        
    # NOte that for NEP-24-024 the second half of the graph is best but the code does not account for it. Alkalinity is ~0     
        
    df['gran'] = gran(df['vol (L)'], df['pH'], v0)

    def find_best_subset_peak(df):
        best_r2 = 0
        best_subset = None
        peak_found = False

        current_subset = df.copy()

        while len(current_subset) >= 5 and not peak_found:
            # Fit the model with the current subset
            X = sm.add_constant(current_subset['vol (L)'])  # Adds a constant term to the predictor
            y = current_subset['gran']
            model = sm.OLS(y, X)
            results = model.fit()

            current_r2 = results.rsquared

            # If the current R-squared is higher than the best so far, update best values
            if current_r2 > best_r2:
                best_r2 = current_r2
                best_subset = current_subset.copy()
            else:
                # If R-squared does not improve, we've peaked
                peak_found = True

            # Remove the point with the lowest volume for the next iteration
            current_subset = current_subset.iloc[1:]

        return best_subset, best_r2


    best_subset, best_r2 = find_best_subset_peak(df)
    X_best = sm.add_constant(best_subset['vol (L)'])  # Adds a constant term to the predictor
    model_best = sm.OLS(best_subset['gran'], X_best)
    results_best = model_best.fit()

    intercept = results_best.params[0]
    coefficient = results_best.params[1]

    # Calculate the x-intercept in µL
    vx = -intercept / coefficient * 1000

    # Calculate the alkalinity in µmol/L
    N = 0.05
    A = vx * N / v0 * 10**3
    
    print(SampleID + ' has an alkalinity of ' + str(A))

    # Add alkalinity, R2, and x-intercept to the output dataframe
    output_i = {'SampleID': SampleID, 'Alkalinity (uM)': A, 'R2': best_r2, 'vx (µL)': vx}
    
    # Convert to dataframe and concatenate with output dataframe
    output = pd.concat([output, pd.DataFrame(output_i, index=[0])], ignore_index=True)

#output = output.sort_values(by output)

# Sort the output dataframe by 'SampleID'
output = output.sort_values(by='SampleID')

# Output the final dataframe
print(output.tail(10))

# Save output to CSV
output.to_excel("data/processed/alkalinity_results.xlsx", index=False)

# Plot Gran titration for the current sample
#plot_gran_titration("NEP-24-035", df, best_subset, intercept, coefficient, A, best_r2, vx)
#Plot Needs work



## If sample has ** in the filename, look back at notes as to why it's wrong...

## Samples that have negative alkalinities are also obviously ** starred, i.e. alkalinity is close to zero

## NEP-24-033** has too flat a line at the start that the intercept is wrong
## NEP-24-061** has too flat a line at the start that the intercept is wrong
## NEP-24-068** has too flat a line at the start that the intercept is wrong

## Sample 044** was titrated in between probes so has an odd shape. In range 30-38uM




sys.exit()





