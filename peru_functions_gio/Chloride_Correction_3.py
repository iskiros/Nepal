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
from peru_functions_gio.NICB_Valid import NICB_Valid
from peru_functions_gio.charge_balance import charge_balance


# A third Chloride Correction document just to see whether blatantly subtracting the Cl values of the rain from the river data will work

######### IT DOES WORK, CHECK NOTES #########



######### CALCULATING MOLAR CONCENTRATIONS FOR THE RAIN DATA #########

######### #Note IT HAS CHANGED FROM THE STANDARD, AS THE SUFFIX IS NOW (PPM) INKEEPING WITH THE EXCEL DOCUMENT #########

def molar_conc_seawater(df):
    """Calculate molar concentration of the major elements"""
    element_dict = {
        'molecules': ['Ca', 'Mg', 'K', 'Na', 'HCO3', 'Cl', 'SO4', 'Al', 'As', 'Ba', 'Bi', 'Ce', 'Fe', 'Li', 'Rb', 'Sr', 'CO3', 'H2PO4'],
        'molar_mass': [40.08, 24.31, 39.10, 22.99, 61.02, 35.45, 96.06, 26.98, 74.92, 137.33, 208.98, 140.12, 55.85, 6.94, 85.47, 87.62, 60.01, 97.99],
        'valency': [2, 2, 1, 1, 1, 1, 2, 3, 3, 2, 3, 3, 2, 1, 1, 2, 2, 1]
    }

    # Suffixes
    suffix = ' (ppm)'
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



def Chloride_Correction_3(df, df2):
    
    ######### PREVIOUS CORRECTION DONE FOR SEAWATER VALUES BEFORE WE HAD RAIN DATA #########
    
    # Seawater data in typical concentrations in ppm
    seawater_data = {
        'Parameter': ['Cl', 'Na', 'SO4', 'Mg', 'Ca', 'K', 'HCO3', 'Sr'],
        'Typical Seawater': [19353, 10693, 2712, 1284, 412, 399, 126, 13]
    }
    
    #ppm (Pretet et al 2014)

    #### 

    # # Convert to DataFrame
    # rain = pd.DataFrame(seawater_data).set_index('Parameter').transpose()

    # # Convert concentrations to molar units (assuming molar_conc_seawater function is defined)
    # rain_processed = molar_conc_seawater(rain)

    # # Ensure columns are in correct order
    # rain_processed = rain_processed.loc[:, ['Ca [aq] (mM)', 'Mg [aq] (mM)', 'K [aq] (mM)', 'Na [aq] (mM)', 'HCO3 [aq] (mM)', 'Cl [aq] (mM)', 'SO4 [aq] (mM)', 'Sr [aq] (mM)']]
    
    # #print(rain_processed)
    
    # # Filter df to include only valid samples
    # df = df[df['unique_code'].isin(df2['unique_code_valid'])]

    # # Create a copy of df for modifications
    # df_copy = df.copy()
    
    ###########################################################################################
    
    
    
    
    ######### RAIN MANIPULATION #########
    
    rain = pd.read_excel('chemweathering/agilentexporttools/Rain_Ions.xlsx', sheet_name='Conconcentration (Processed)')
    
    rain = rain.loc[rain['Label'] == 'Refugio V. Rainwater'] # so far the best charge balanced one
    
    # Convert concentrations to molar units (assuming molar_conc_seawater function is defined)
    rain_processed = molar_conc_seawater(rain)
    
    ## Change the first column, first row name to 'unique_code'
    rain_processed.columns = ['unique_code'] + [col for col in rain_processed.columns[1:]]
    
    ####################################
    
    
    
    
    ######### CHARGE BALANCE #########
    
    # Charge balance
    rain_processed = charge_balance(rain_processed)

    # Initialize lists to store unique codes and NICB values 
    NICB_Balance = NICB_Valid(rain_processed)
    
    ####################################
    
    
    ######### ION PROCESSING #########

    # Ensure columns are in correct order
    rain_processed = rain_processed.loc[:, ['Ca [aq] (mM)', 'Mg [aq] (mM)', 'K [aq] (mM)', 'Na [aq] (mM)', 'Cl [aq] (mM)', 'SO4 [aq] (mM)', 'Sr [aq] (mM)']]
    
    # note that there is no 'HCO3 [aq] (mM)' in the rain_ions document. REFER TO CHLORIDE CORRECTION 2
    

    
    ######### FILTERING AND CORRECTION #########
    
    # Filter df to include only valid samples
    df = df[df['unique_code'].isin(df2['unique_code_valid'])]
    
    # Remove row with string "T1BWF-1121" in the unique_code column
    df = df[~df['unique_code'].str.contains('T1BWF-1121', na=False)]

    # Create a copy of df for modifications
    df_copy = df.copy()
    
    #cl_minimum = the lowest cl value in df_copy:
    cl_minimum = df_copy['Cl [aq] (mM)'].min()
    
    ###########################################



    ######### SUBTRACTION CALCULATION #########
    
    # Iterate over each element (column) in the rain_processed DataFrame
    for element in rain_processed.columns:
    # Check if the current element exists as a column in the df_copy DataFrame
    
        if element in df_copy.columns:
        # Get the value of the current element from the rain_processed DataFrame (assuming there's only one value)
        
            rain_element_value = rain_processed[element].values[0]
        
        
        # Get the value of 'Cl [aq] (mM)' from the rain_processed DataFrame (assuming there's only one value)
        
            cl_rain_value = rain_processed['Cl [aq] (mM)'].values[0]
            
            cl_minimum_star = cl_minimum - cl_rain_value
            
            #print('Cl minimum star:',
            #    cl_minimum_star)  
            
            
            
        
        # Check to ensure the Cl value is not zero to avoid division by zero
            if cl_rain_value != 0:
                if cl_minimum_star >= 0:
                    # Perform the correction calculation:
                    # For each row in df_copy, subtract the scaled rain element value from the river element value
                    df_copy['*' + element] = df_copy[element] - rain_element_value 
                else:
                    # Set cl_minimum_star to zero
                    cl_minimum_star = 0
                    # Perform the correction calculation with cl_minimum_star set to zero
                    df_copy['*' + element] = df_copy[element] - rain_element_value
            

        else:
        # Print a message if the current element is not found in the df_copy DataFrame
            print(f"Element {element} not found in df_copy")
            
            
    #############################################
    
    
    
    ######### PLOTTING RESULTS #########        
            
    ## make a plot of the Cl* values in df_copy:
    plt.figure(figsize=(10,6))
    plt.scatter(df_copy['altitude_in_m'], df_copy['*Cl [aq] (mM)'], alpha=0.7, s=70)
    plt.ylabel('*Cl [aq] (mM)')
    plt.xlabel('Altitude (m)')
    plt.title('Scatter plot of *Cl vs. Altitude')
    #plt.savefig('chemweathering/figures/Clstar.png')
    #plt.show()
    plt.close()

    ####################################
    
    
    ######### FINAL OUTPUT FOR SHOWING #########
            

    # Select relevant columns for final output
    df_copy = df_copy[['unique_code'] + [col for col in df_copy.columns if '*' in col] + ['latitude_converted', 'longitude_converted', 'NICB', 'calculated_discharge_in_m3_s-1', 'Alkalinity_CaCO3_in_mg_kg-1', 'altitude_in_m', 'water_body']]

    return df_copy

