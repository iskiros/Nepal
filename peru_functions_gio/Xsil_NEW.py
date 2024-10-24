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
# - From when I was investigating fudging the numbers to calculate a minimum amount of Cl for evaporite deposits
# - Not used anymore



def rotate_points_180(df_lons, df_lats, center_lon, center_lat):
    # Rotate points by 180 degrees around the center
    rotated_lons = 2 * center_lon - df_lons
    rotated_lats = 2 * center_lat - df_lats
    return rotated_lons, rotated_lats



def Xsil_NEW(df, mg_na_sil_ratio, ca_na_sil_ratio, num_simulations=10000, cl_evap_min=0, cl_evap_max=10.0):
    df_copy = df.copy()
    
    # Filter out specific unique_codes
    df_copy = df_copy[~df_copy['unique_code'].str.contains('T1DW-0322', na=False)]
    df_copy = df_copy[~df_copy['unique_code'].str.contains('RC15WF-1121', na=False)]
    
    # Initialize arrays to store results
    optimal_cl_evap_values = []
    optimal_X_Sil_values = []
    
    # Define numerator factor
    numerator_factor = 1 + 2 * mg_na_sil_ratio + 2 * ca_na_sil_ratio
    
    for index, row in df_copy.iterrows():
        best_cl_evap = None
        best_X_Sil = -np.inf
        
        for _ in range(num_simulations): # "_" is used when the loop variable is not used
            cl_evap = np.random.uniform(cl_evap_min, cl_evap_max)
            na_sil = row['*Na [aq] (mM)'] - cl_evap
            
            # These two lines generate a random value for cl_evap using np.random.uniform within the range specified by cl_evap_min and cl_evap_max. [set to 0 and 10.0 respectively] 
            # Then, it calculates the value of na_sil by subtracting cl_evap from the value of the '*Na [aq] (mM)' column in the current row.
            
            if na_sil < 0:
                continue
            numerator = (na_sil * numerator_factor) + row['*K [aq] (mM)']
            denominator = row['*Na [aq] (mM)'] + row['*K [aq] (mM)'] + 2 * row['*Mg [aq] (mM)'] + 2 * row['*Ca [aq] (mM)']
            x_sil = numerator / denominator
            
            if x_sil <= 1 and x_sil > best_X_Sil:
                best_X_Sil = x_sil
                best_cl_evap = cl_evap
                # This block of code checks if x_sil is less than or equal to 1 and greater than the current best_X_Sil. 
                # If it is, it updates best_X_Sil with the new value of x_sil and assigns the current cl_evap to best_cl_evap.
        
        optimal_cl_evap_values.append(best_cl_evap)
        optimal_X_Sil_values.append(best_X_Sil)
    
    # Store optimal values in the DataFrame
    df_copy['optimal_cl_evap'] = optimal_cl_evap_values
    df_copy['X_Sil_NEW'] = optimal_X_Sil_values
    
    ## Plot the optimal values of cl_evap and X_Sil:
    
    df_copy.dropna(subset=['optimal_cl_evap', 'X_Sil_NEW'], inplace=True)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df_copy['optimal_cl_evap'], df_copy['X_Sil_NEW'], alpha=0.7, s=70)
    plt.xlabel('cl_evap')
    plt.ylabel('X_Sil_NEW')
    plt.title('Scatter plot of cl_evap vs. X_Sil_NEW')
    

    # Annotate each point with unique_code, skipping rows where unique_code is 'nan'
    for index, row in df_copy.iterrows():
        plt.text(row['optimal_cl_evap'], row['X_Sil_NEW'], row['unique_code'], fontsize=8, ha='center', va='bottom')

    plt.savefig('chemweathering/figures/cl_evap_vs_XSil.png')
    plt.show()
    plt.close()
    
    return df_copy
