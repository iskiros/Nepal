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





def molar_conc_seawater(df):
    """Calculate molar concentration of the major elements"""
    element_dict = {
        'molecules': ['Ca', 'Mg', 'K', 'Na', 'HCO3', 'Cl', 'SO4', 'Al', 'As', 'Ba', 'Bi', 'Ce', 'Fe', 'Li', 'Rb', 'Sr', 'CO3', 'H2PO4'],
        'molar_mass': [40.08, 24.31, 39.10, 22.99, 61.02, 35.45, 96.06, 26.98, 74.92, 137.33, 208.98, 140.12, 55.85, 6.94, 85.47, 87.62, 60.01, 97.99],
        'valency': [2, 2, 1, 1, 1, 1, 2, 3, 3, 2, 3, 3, 2, 1, 1, 2, 2, 1]
    }

    # Suffixes
    suffix = ''
    alt_suffix = ''

    df_copy = df.copy()

    for element, molar_mass, valency in zip(element_dict['molecules'],
                                            element_dict['molar_mass'],
                                            element_dict['valency']):
        # Check if the standard or alternate column exists
        if element + suffix in df_copy.columns:
            col_name = element + suffix
        elif element + alt_suffix in df_copy.columns:
            col_name = element + alt_suffix
        else:
            # Skip element if neither column exists
            continue

        # Create new molar columns
        df_copy.loc[:, element + ' [aq] (mM)'] = df_copy[col_name] / molar_mass
        df_copy.loc[:, element + ' [aq] (meq/L)'] = df_copy.loc[:, element + ' [aq] (mM)'] * valency

    #print(df_copy.columns)

    # Select relevant columns
    #df_copy = [ [col for col in df_copy.columns if ' [aq] (mM)' in col] + [col for col in df_copy.columns if ' [aq] (meq/L)' in col] ]

    #print(df_copy)

    return df_copy



def rotate_points_180(df_lons, df_lats, center_lon, center_lat):
    # Rotate points by 180 degrees around the center
    rotated_lons = 2 * center_lon - df_lons
    rotated_lats = 2 * center_lat - df_lats
    return rotated_lons, rotated_lats


########################### OLD SIL CORRECTION TO FUDGE THE NUMBERS FOR THE RATIOS ###########################
# - Function made when we were still investigating our ratios being wrong. 
# This basically simulates and fudges the numbers so that the XSil is <= 1 for particular ratios.
# - Code takes long and is unsuccessful




def Correction_Sil(df, df2):
    # Seawater data in typical concentrations in ppm
    seawater_data = {
        'Parameter': ['Cl', 'Na', 'SO4', 'Mg', 'Ca', 'K', 'HCO3', 'Sr'],
        'Typical Seawater': [19353, 10693, 2712, 1284, 412, 399, 126, 13]
    }
    
    #ppm (Pretet et al 2014)

    # Convert to DataFrame
    rain = pd.DataFrame(seawater_data).set_index('Parameter').transpose()

    # Convert concentrations to molar units (assuming molar_conc_seawater function is defined)
    rain_processed = molar_conc_seawater(rain)

    # Ensure columns are in correct order
    rain_processed = rain_processed.loc[:, ['Ca [aq] (mM)', 'Mg [aq] (mM)', 'K [aq] (mM)', 'Na [aq] (mM)', 'HCO3 [aq] (mM)', 'Cl [aq] (mM)', 'SO4 [aq] (mM)', 'Sr [aq] (mM)']]
    
    #print(rain_processed)
    
    # Filter df to include only valid samples
    df = df[df['unique_code'].isin(df2['unique_code_valid'])]
    
    #### THIS takes on NICB Balanced samples, which is the df2 dataframe
    

    # Create a copy of df for modifications
    df_copy = df.copy()
    
    #cl_dilute = the lowest cl value in df_copy:
    cl_dilute = df_copy['Cl [aq] (mM)'].min()
    
    # Initialize arrays to store results
    optimal_X_Sil_values = []
    
    # Initialize arrays to store results
    optimal_X_Sil_values = []
    optimal_mg_na_sil_values = []
    optimal_cl_dilute_values = []
    optimal_ca_na_sil_values = []

    # Iterate over each row in df_copy
    for index, row in df_copy.iterrows():
        # Initialize the best values for this row
        best_X_Sil = -np.inf
        best_mg_na_sil = None
        best_cl_dilute = None
        best_ca_na_sil = None
        
        # Set the number of simulations
        num_simulations = 10  # Increase the number of simulations for more iterations
        
        # Iterate over cl_dilute values from the lowest value to zero
        for cl_dilute_value in np.linspace(cl_dilute, 0, num=num_simulations):
            # Iterate over Mg/Na ratios
            for mg_na_sil_ratio in np.linspace(0, 1, num=num_simulations):
                # Iterate over Ca/Na ratios
                for ca_na_sil_ratio in np.linspace(0, 1, num=num_simulations):
                    # Create a copy of the current row for modifications
                    row_copy = row.copy()
                    
                    # Define numerator factor
                    numerator_factor = 1 + 2 * mg_na_sil_ratio + 2 * ca_na_sil_ratio
                    
                    # Correction calculation
                    for element in rain_processed.columns:
                        if element in row_copy.index:
                            rain_element_value = rain_processed[element].values[0]
                            cl_rain_value = rain_processed['Cl [aq] (mM)'].values[0]
                            
                            if cl_rain_value != 0:
                                row_copy['*' + element] = row_copy[element] - ((rain_element_value / cl_rain_value) * (row_copy['Cl [aq] (mM)'] - cl_dilute_value))
                    
                    # Calculate X_Sil_NEW
                    na_sil = row_copy['*Na [aq] (mM)']
                    numerator = (na_sil * numerator_factor) + row_copy['*K [aq] (mM)']
                    denominator = row_copy['*Na [aq] (mM)'] + row_copy['*K [aq] (mM)'] + 2 * row_copy['*Mg [aq] (mM)'] + 2 * row_copy['*Ca [aq] (mM)']
                    x_sil = numerator / denominator
                    
                    if x_sil <= 1 and x_sil > best_X_Sil:
                        best_X_Sil = x_sil
                        best_mg_na_sil = mg_na_sil_ratio
                        best_cl_dilute = cl_dilute_value
                        best_ca_na_sil = ca_na_sil_ratio
        
        # Store the best values for this row
        optimal_X_Sil_values.append(best_X_Sil)
        optimal_mg_na_sil_values.append(best_mg_na_sil)
        optimal_cl_dilute_values.append(best_cl_dilute)
        optimal_ca_na_sil_values.append(best_ca_na_sil)

    # Append the optimal values to the DataFrame
    df_copy['optimal_X_Sil'] = optimal_X_Sil_values
    df_copy['optimal_mg_na_sil'] = optimal_mg_na_sil_values
    df_copy['optimal_cl_dilute'] = optimal_cl_dilute_values
    df_copy['optimal_ca_na_sil'] = optimal_ca_na_sil_values
    # Plot the optimal values with unique codes
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df_copy['optimal_mg_na_sil'], df_copy['optimal_ca_na_sil'], c=df_copy['optimal_X_Sil'], cmap='viridis', alpha=0.7, s=70)
    plt.colorbar(scatter, label='X_Sil_NEW')
    
        # Annotate each point with unique_code, skipping rows where unique_code is 'nan'
    for index, row in df_copy.iterrows():
        plt.text(row['optimal_mg_na_sil'], row['optimal_ca_na_sil'], row['unique_code'], fontsize=8, ha='center', va='bottom')

    plt.xlabel('Mg/Na ratio')
    plt.ylabel('Ca/Na ratio')
    plt.title('Scatter plot of Mg/Na ratio vs. Ca/Na ratio with X_Sil_NEW')
    plt.savefig('chemweathering/figures/optimal_values_plot.png')
    plt.show()
    plt.close()
        

    
    
    return df_copy



