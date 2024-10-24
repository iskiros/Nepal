import sys
import os
import math
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geodatasets
import geopandas as gpd
import re
import folium
from folium import plugins
import earthpy as et
import webbrowser
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import get_cmap
from branca.colormap import LinearColormap
import statsmodels.api as sm
from scipy.stats import norm
import elevation
import shapely.geometry
from shapely.geometry import Point, box, Polygon, MultiPolygon
import seaborn as sns
from shapely.wkt import loads
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
import rasterio
from rasterio.plot import show
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import from_bounds, from_origin
from rasterio.mask import mask
import earthpy.spatial as es
from scipy.interpolate import RegularGridInterpolator
import matplotlib.colors as mcolors


########################### FUNCTION DEFINITION ###########################
# - Plots Na/Cl and Mg/Cl against Na/Cl for not corrected values
# - Used to calculate a rain endmember Na/Cl value for our samples, which fits with calculated data
# - Also used to investigate HCO3/Cl rain ratio. Have since then used seawater



def uncorrected_plots(df, df2):
    
    
    ########## DATA FRAME MAINTENANCE ##########
    
    df_copy = df.copy()
    
    # Filter df to include only valid samples
    df_copy = df_copy[df_copy['unique_code'].isin(df2['unique_code_valid'])]
    
    ## Remove RC15WF-1121:
    df_copy = df_copy[~df_copy['unique_code'].str.contains('RC15WF-1121', na=False)]
    
    ## Now should have "df_neat"
    
    df_copy['Na/Cl'] = df_copy['Na [aq] (mM)']/df_copy['Cl [aq] (mM)']  
    
    df_copy['Mg/Cl'] = df_copy['Mg [aq] (mM)']/df_copy['Cl [aq] (mM)']
    
    df_copy['HCO3/Cl'] = df_copy['HCO3 [aq] (mM)']/df_copy['Cl [aq] (mM)']
    
    ##################################################
    
    
    
    ########## PLOT NA/CL AND MG/CL ##########
    
    # Plot Mg/Cl against Na/Cl both on a log scale
    fig, ax = plt.subplots()
    #ax.scatter(df_copy['Na/Cl'], df_copy['Mg/Cl'], c='blue')

    ax.title.set_text('Mg/Cl vs Na/Cl')
    # log scale both axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    
    
    #for index, row in df_copy.iterrows():
    #    ax.text(row['Na/Cl'], row['Mg/Cl'], row['unique_code'])
        
    # differentiate between tributaries and river samples
    markers = {'mainstream': 'o', 'tributary': '^', 'spring': '*'}


    for water_body, marker in markers.items():
        subset = df_copy[df_copy['water_body'] == water_body]
        ax.scatter(
            subset['Na/Cl'].values,
            subset['Mg/Cl'].values,
            alpha=0.9,
            color =  'blue',
            edgecolor = 'black',
            marker=marker,
            linewidth=1
        )

        
        
    # Create legend with custom markers and labels
    for water_body, marker in markers.items():
        ax.scatter([], [], marker=marker, color='blue', label=water_body)     
        
        
    ##################################################
    
    ##################################################
    
    ########## BECAUSE IT'S A LOG PLOT, NEED TO FACTOR THAT IN WHEN FITTING THE LINE    
        
       # Filter and drop NaN values for 'mainstream' samples
    nacl_values = df_copy[df_copy['water_body'] == 'mainstream']['Na/Cl'].dropna()
    mgcl_values = df_copy[df_copy['water_body'] == 'mainstream']['Mg/Cl'].dropna()

    # Convert to numpy arrays and ensure there are no invalid values for log
    nacl_values = nacl_values[nacl_values > 0].to_numpy()
    mgcl_values = mgcl_values[mgcl_values > 0].to_numpy()

    # Check if na_values and cl_values are valid for np.log
    if len(nacl_values) == 0 or len(mgcl_values) == 0:
        raise ValueError("No valid values for Na or Cl after filtering non-positive values.")

    # Ensure na_values and cl_values are arrays of appropriate dtype
    nacl_values = np.array(nacl_values, dtype=float)
    mgcl_values = np.array(mgcl_values, dtype=float)

    # Take the logarithm of the values
    x = np.log(nacl_values).reshape(-1, 1)
    y = np.log(mgcl_values).reshape(-1, 1)
    
    
    
    ########## FITTING THE LINE ##########
    
    # Fit the model
    model = sm.OLS(y, sm.add_constant(x)).fit()
    predictions = model.predict(sm.add_constant(x))

    # Plot the best fit line
    ax.plot(np.exp(x), np.exp(predictions), c='orange', linewidth=2)       
        
    ########################################
    
    
    
    ########## TORRES ET AL, 2015 ##########    
        
    ## add a box between x = 0.93 and 2.17, and y ranging from 0.2 to 0.9: Torres Rain
    x = np.linspace(0.93, 2.17, 100)
    y = np.linspace(0.2, 0.2, 100)
    ax.plot(x, y, c='red', linewidth=2)
    
    ## add a box between x = 0.93 and 2.17, and y ranging from 0.2 to 0.9: Torres Rain
    x = np.linspace(0.93, 2.17, 100)
    y = np.linspace(0.9, 0.9, 100)
    ax.plot(x, y, c='red', linewidth=2)
    
    ## add a box between x = 0.93 and 0.93, and y ranging from 0.2 to 0.9: Torres Rain
    x = np.linspace(0.93, 0.93, 100)
    y = np.linspace(0.2, 0.9, 100)
    ax.plot(x, y, c='red', linewidth=2)
    
    ## add a box between x = 2.17 and 2.17, and y ranging from 0.2 to 0.9: Torres Rain
    x = np.linspace(2.17, 2.17, 100)
    y = np.linspace(0.2, 0.9, 100)
    ax.plot(x, y, c='red', linewidth=2)
    
    
    ########################################
    
    
    
    ########## Torres samples and gaillardet evaporite endmember below ##########

    
    # ## add a line between x = 10 and 100, and y ranging from 10 to 100: Torres Samples
    # x = np.linspace(10, 100, 100)
    # y = np.linspace(10, 100, 100)
    # ax.plot(x, y, c='green', linewidth=2)
    
    
    # ## add a line between x = 1, and y ranging from 0.008–0.03:
    # x = np.linspace(1, 1, 100)
    # y = np.linspace(0.008, 0.03, 100)
    # ax.plot(x, y, c='brown', linewidth=2)
    
    
    ############################################################
    
    
    ########## NOT ADDED TORRES SAMPLES AND GAILLARDET EVAPORITE ENDMEMBER ##########
    
    # Add legend for points
    #ax.scatter([], [], c='blue', label='Our Samples')

    # Add legend for line for Torres Rain
    #ax.plot([], [], c='red', linewidth=2, label='Torres Rain fit') #Fudging it
    
    # Add legend for line for Torres Samples
    #ax.plot([], [], c='green', linewidth=2, label='Torres Samples fit') #Fudging it
    
    
    ############################## OUR RAIN AND RIVER SAMPLES ################################
    
    # Add legend for line for our rain samples
    ax.plot([], [], c='red', linewidth=2, label='Rain Samples') #Fudging it
    
    # Add legend for line for our rain samples
    ax.plot([], [], c='orange', linewidth=2, label='Best Fit River Samples') #Fudging it
    
    ################################################################################
    
    
    # Add legend for line for Gaillardet Evaporite 
    #ax.plot([], [], c='brown', linewidth=2, label='Gaillardet Evaporite Endmember') #Fudging it
    
    ## Plot a large box point at 1,1
    #ax.scatter(1, 1, c='red', s=100, marker='o')
    

    # Display legend
    ax.legend()
    
    
    
    ax.set_xlabel('Na/Cl')
    ax.set_ylabel('Mg/Cl')
    
    #plt.show()
    plt.close()
    
    
    
    
    ############################################################
    
    
    ############### HCO3/CL RATIO INVESTIGATION ################
    
    ############################################################
    
    # Plot HCO3/Cl against Na/Cl both on a log scale
    fig, ax = plt.subplots()
    #ax.scatter(df_copy['Na/Cl'], df_copy['Mg/Cl'], c='blue')

    ax.title.set_text('HCO3/Cl vs Na/Cl')
    # log scale both axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    
    
    #for index, row in df_copy.iterrows():
    #    ax.text(row['Na/Cl'], row['Mg/Cl'], row['unique_code'])
        
    # differentiate between tributaries and river samples
    markers = {'mainstream': 'o', 'tributary': '^', 'spring': '*'}


    for water_body, marker in markers.items():
        subset = df_copy[df_copy['water_body'] == water_body]
        ax.scatter(
            subset['Na/Cl'].values,
            subset['HCO3/Cl'].values,
            alpha=0.9,
            color =  'blue',
            edgecolor = 'black',
            marker=marker,
            linewidth=1
        )

        
        
    # Create legend with custom markers and labels
    for water_body, marker in markers.items():
        ax.scatter([], [], marker=marker, color='blue', label=water_body)     
        
        
       # Filter and drop NaN values for 'mainstream' samples
    nacl_values = df_copy[df_copy['water_body'] == 'mainstream']['Na/Cl'].dropna()
    hco3cl_values = df_copy[df_copy['water_body'] == 'mainstream']['HCO3/Cl'].dropna()

    # Convert to numpy arrays and ensure there are no invalid values for log
    nacl_values = nacl_values[nacl_values > 0].to_numpy()
    hco3cl_values = hco3cl_values[hco3cl_values > 0].to_numpy()

    # Check if na_values and cl_values are valid for np.log
    if len(nacl_values) == 0 or len(hco3cl_values) == 0:
        raise ValueError("No valid values for Na or Cl after filtering non-positive values.")

    # Ensure na_values and cl_values are arrays of appropriate dtype
    nacl_values = np.array(nacl_values, dtype=float)
    hco3cl_values = np.array(hco3cl_values, dtype=float)

    # Take the logarithm of the values
    x = np.log(nacl_values).reshape(-1, 1)
    y = np.log(hco3cl_values).reshape(-1, 1)
    
    #############################################
    
    ############### FITTING #####################
    
    
    # Fit the model
    model = sm.OLS(y, sm.add_constant(x)).fit()
    predictions = model.predict(sm.add_constant(x))

    # Plot the best fit line
    ax.plot(np.exp(x), np.exp(predictions), c='orange', linewidth=2)       
        
        
    ## add a line at x = 0.93
    ax.axvline(x=0.93, c='red', linewidth=2)
    
    ## add a line at x = 2.17
    ax.axvline(x=2.17, c='red', linewidth=2)
    
    ## add a line at the average of the two
    ax.axvline(x=1.55, c='green', linewidth=2)
    
    
    ### See where x = 1.55 intersects with the best fit line
    intercept1 = model.params[0]
    slope1 = model.params[1]
    
    x_intersect1 = 1.55
    y_intersect1 = slope1*x_intersect1 + intercept1
    
    print('Y intercept for HCO3/Cl is: ', y_intersect1)
    
    

    x_intersect2 = 0.93
    y_intersect2 = slope1*x_intersect2 + intercept1
    
    print('LOWER BOUND Y intercept for HCO3/Cl is: ', y_intersect2)
    
    
    
    
    x_intersect3 = 2.17
    y_intersect3 = slope1*x_intersect3 + intercept1
    
    print('UPPER BOUND Y intercept for HCO3/Cl is: ', y_intersect3)
    
    ############################################################
    
    
    ############################################################
    
    
    
    # Add legend for line for our rain samples
    ax.plot([], [], c='red', linewidth=2, label='Rain Samples') #Fudging it
    
    # Add legend for line for our rain samples
    ax.plot([], [], c='orange', linewidth=2, label='Best Fit River Samples') #Fudging it
    
    # Add legend for line for Gaillardet Evaporite 
    #ax.plot([], [], c='brown', linewidth=2, label='Gaillardet Evaporite Endmember') #Fudging it
    
    ## Plot a large box point at 1,1
    #ax.scatter(1, 1, c='red', s=100, marker='o')
    

    # Display legend
    ax.legend()
        
        
    
    ax.set_xlabel('Na/Cl')
    ax.set_ylabel('HCO3/Cl')
    
    #plt.show()
    plt.close()
    
    
    
    
    
    
    ##################
    

    
    ## What is the lowest Cl value?
    print('Lowest Cl value is:', df_copy['Cl [aq] (mM)'].min())
    
    ## Remove all samples with Cl = minimum
    #df_copy_purged = df_copy[df_copy['Cl [aq] (mM)'] != df_copy['Cl [aq] (mM)'].min()]
    df_copy_purged = df_copy
    
    
    
    ### Plot Na vs Cl for all samples on a log-log plot
    
    fig, ax = plt.subplots()
    #ax.scatter(df_copy['Na [aq] (mM)'], df_copy['Cl [aq] (mM)'], c='blue')
    ax.title.set_text('Na vs Cl, all samples, uncorrected')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    for index, row in df_copy.iterrows():
        ax.text(row['Na [aq] (mM)'], row['Cl [aq] (mM)'], row['unique_code'])
    
    # differentiate between tributaries and river samples
    markers = {'mainstream': 'o', 'tributary': '^', 'spring': '*'}
    
    vmin1, vmax1 = df_copy_purged['altitude_in_m'].min(), df_copy_purged['altitude_in_m'].max()
    for water_body, marker in markers.items():
        subset = df_copy_purged[df_copy_purged['water_body'] == water_body]
        sc = ax.scatter(
            subset['Na [aq] (mM)'].values,
            subset['Cl [aq] (mM)'].values,
            alpha=0.9,
            # colour them by altitude:
            c=subset['altitude_in_m'],
            edgecolor='black',
            marker=marker,
            linewidth=1,
            vmin = vmin1,
            vmax = vmax1
        )
        
    # Add colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label('Altitude (m)')

            
            
    # Create legend with custom markers and labels
    for water_body, marker in markers.items():
        ax.scatter([], [], marker=marker, color='blue', label=water_body)
        

    # Filter and drop NaN values for 'mainstream' samples
    na_values = df_copy[df_copy['water_body'] == 'mainstream']['Na [aq] (mM)'].dropna()
    cl_values = df_copy[df_copy['water_body'] == 'mainstream']['Cl [aq] (mM)'].dropna()

    # Convert to numpy arrays and ensure there are no invalid values for log
    na_values = na_values[na_values > 0].to_numpy()
    cl_values = cl_values[cl_values > 0].to_numpy()

    # Check if na_values and cl_values are valid for np.log
    if len(na_values) == 0 or len(cl_values) == 0:
        raise ValueError("No valid values for Na or Cl after filtering non-positive values.")

    # Ensure na_values and cl_values are arrays of appropriate dtype
    na_values = np.array(na_values, dtype=float)
    cl_values = np.array(cl_values, dtype=float)

    # Take the logarithm of the values
    x = np.log(na_values).reshape(-1, 1)
    y = np.log(cl_values).reshape(-1, 1)
    
    
    
    # Fit the model
    model = sm.OLS(y, sm.add_constant(x)).fit()
    predictions = model.predict(sm.add_constant(x))

    # Plot the best fit line
    #ax.plot(np.exp(x), np.exp(predictions), c='red', linewidth=2)
    
    # Print y = mx + c
    #print('y =', model.params[1], 'x +', model.params[0])
    
    
    
    
    
    # Calculate parameters for the purple line in log-log space
    cl_min = df_copy_purged['Cl [aq] (mM)'].min()
    gradient = model.params[1]
    na_min = (cl_min - model.params[0]) / gradient

    # Transform the endpoints to log scale
    log_na_min = np.log(na_min)
    log_cl_min = np.log(cl_min)

    # Define the line in log-log space
    log_x = np.linspace(np.log(0.01), log_na_min, 100)
    log_y = np.linspace(np.log(0.01), log_cl_min, 100)

    # Plot the line in log-log space
    #ax.plot(np.exp(log_x), np.exp(log_y), c='purple', linewidth=2)
    
    na_cl_mainstream = 1/gradient
    

    intercept_red = model.params[0]
    slope_red = model.params[1]


    
    # Parameters for the purple line
    cl_min = df_copy_purged['Cl [aq] (mM)'].min()
    na_min = ((cl_min - intercept_red)) / slope_red

    # Transform the endpoints to log scale
    log_na_min = np.log(na_min)
    log_cl_min = np.log(cl_min)

    # Slope of the purple line in log-log space
    slope_purple = (log_cl_min - np.log(0.01)) / (log_na_min - np.log(0.01))
    
    #Print Na/Cl of purple line
    print('Na/Cl of purple line is:', 1/slope_purple)

    # Intercept of the purple line in log-log space
    intercept_purple = log_cl_min - slope_purple * log_na_min

    # Solve for the intersection point in log-log space
    log_x_intersect = (intercept_purple - intercept_red) / (slope_red - slope_purple)
    log_y_intersect = slope_red * log_x_intersect + intercept_red

    # Transform back to linear space
    x_intersect = np.exp(log_x_intersect)
    y_intersect = np.exp(log_y_intersect)
    
    

    

    
    #ax.scatter(x_intersect, y_intersect, c='Orange', s=100, marker='x')

    #ax.plot([], [], c='Orange', linewidth=2, label='Intersection point')


    gradient = model.params[1]
    
    na_cl_mainstream = 1/gradient


    
    # Add a 1:1 Na:Cl line
    x = np.linspace(0.01, 4, 100)
    y = np.linspace(0.01, 4, 100)
    ax.plot(x, y, c='green', linewidth=2)    
    
    # Add legend for the 1:1 line
    ax.plot([], [], c='green', linewidth=2, label='1:1 Na:Cl')
    
    # Add legend for the mainstream line, rounded to 2dp
    #ax.plot([], [], c='red', linewidth=2, label='Mainstream, Na:Cl = ' + str(round(na_cl_mainstream, 3)))
    
    # Add legend for the purple line. This was used before
    #ax.plot([], [], c='purple', linewidth=2, label='0,0 to Mainstream')
    
    # Display legend
    ax.legend()
    ax.set_xlabel('Na [aq] (mM)')
    ax.set_ylabel('Cl [aq] (mM)')
    plt.show()
    plt.close()


