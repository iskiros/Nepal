import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import itertools

SampleID = "NEP-24-008"

v1 = np.array([0, 10,15, 20, 25, 30, 36, 51]) /1000 # Volumes are already in the desired format (mL)
pH = np.array([5.92, 4.50, 4.13, 4.08, 4.03, 3.98, 3.98, 3.77])
v = v1 * 1.2







# Compute the Gran function
Vo = 50
values = pd.DataFrame({'v': v, 'pH': pH})
values['gran'] = (Vo + values['v']) * 10 ** -values['pH']



def find_best_subset(df):
    best_r2 = 0
    best_subset = None
    
    # Loop over all possible subsets of at least 5 points
    for i in range(len(df) - 4  ):
        for j in range(i+4, len(df)):
            subset = df.iloc[i:j+1]
            X = subset['v']
            X = sm.add_constant(X)  # Adds a constant term to the predictor
            y = subset['gran']
            model = sm.OLS(y, X)
            results = model.fit()
            
            r2 = results.rsquared
            
            if r2 > best_r2:
                best_r2 = r2
                best_subset = subset
                
    return best_subset, best_r2

filtered_values = values[(values['pH'] <= 5) & (values['pH'] >= 3.35)]


best_subset, best_r2 = find_best_subset(filtered_values)
X_best = sm.add_constant(best_subset['v'])  # Adds a constant term to the predictor
model_best = sm.OLS(best_subset['gran'], X_best)
results_best = model_best.fit()


intercept = results_best.params[0]
coefficient = results_best.params[1]

# Calculate the x-intercept in µL
vx = -intercept / coefficient * 1000

# Calculate the alkalinity in µmol/L
N = 0.05
A = vx * N / Vo * 10**3

# Add alkalinity to the data frame as a new column
values['Alkalinity'] = A

# Save to Excel
excel_filename = f"/Users/enrico/PeruRiversProject/nepal/Excel Files/{SampleID}_titration_table_(v in mL).xlsx"  # <-- Updated
#values.to_excel(excel_filename, index=False)

gran_marker_color = 'black'
pH_marker_color = 'black'


# Plotting
fig, ax1 = plt.subplots(figsize=(10, 7))

# Gran vs Volume of Acid Added
sns.lineplot(x='v', y='gran', data=values, ax=ax1, color='blue', label='Gran Value')
sns.lineplot(x='v', y='gran', data=best_subset, ax=ax1, color='red')

sns.scatterplot(x='v', y='gran', data=values, ax=ax1, color=gran_marker_color, marker='x', s=60, alpha=1, linewidth=1.5)

# Calculate the x-value where the regression line hits y=0
x_zero = -intercept / coefficient

# Determine the x-values for the line extension: starting from the last point of the best subset to x_zero
x_extension = np.linspace(best_subset['v'].iloc[-1], x_zero)

# Calculate the y-values for these x-values using the regression equation
y_extension = intercept + coefficient * x_extension

# Plot the extension
ax1.plot(x_extension, y_extension, '--', color='red')
# Add vertical grid lines
ax1.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)


ax1.set_xlabel('Volume of Acid Added (mL)')
ax1.set_ylabel('Gran Value', color='blue', position=(0, 0.5))
ax1.tick_params('y', colors='blue')
legend = ax1.legend(loc='upper left')
legend.set_bbox_to_anchor((0, 0.85))  # Adjust the second value to move the legend up or down

ax1.set_title(f'Gran Titration - {SampleID}', fontsize=11)  # <-- Updated

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
sns.lineplot(x='v', y='pH', data=values, ax=ax2, color='green', label='pH')
sns.scatterplot(x='v', y='pH', data=values, ax=ax2, color=pH_marker_color, marker='+', s=60, alpha=1, linewidth=1.5)


ax2.set_ylabel('pH', color='green')
ax2.tick_params('y', colors='green')
# Assuming you've created the plots and legends already
legend_ph = ax2.legend(loc='upper left')
legend_ph.set_bbox_to_anchor((0, 0.77))  # Adjust the second value to move the legend up or down


# Display the regression equation and alkalinity on the graph
equation = f"Alkalinity = {A:.2f} \u03BCmol \nR² = {best_r2:.2f} \nVx = {(vx/1000):.2f} ml"




#equation = f"y = {coefficient:.2f}x + {intercept:.2f}\nR² = {best_r2:.2f}\nAlkalinity = {A:.2f}"
props = dict(boxstyle='round,pad=0.3', facecolor='yellow', edgecolor='black')
ax1.text(0.015, 0.67, equation, transform=ax1.transAxes, verticalalignment='top', bbox=props)

fig.tight_layout()  # to make sure the right y-label is not slightly clipped
plot_filename = "/Users/enrico/PeruRiversProject/nepal/Plots/Gran_plot_" + SampleID + ".png"
#plt.savefig(plot_filename, dpi=300)
plt.show()