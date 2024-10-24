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



########################### FUNCTION DEFINITION ###########################
# - From when we were trying to see how the values would stack up if you plotted the values for XSil uncorrected, 
# to back calculate what ratios were required and what evaporite comp was required for Xsil<1
# - Not used anymore



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


    return df_copy



def Chloride_XSil_Correction(df, df2):
    
    rain = pd.read_excel('chemweathering/agilentexporttools/Rain_Ions.xlsx', sheet_name='Conconcentration (Processed)')
    
    rain = rain.loc[rain['Label'] == 'Refugio V. Rainwater'] # so far the best charge balanced one
    
    # Convert concentrations to molar units (assuming molar_conc_seawater function is defined)
    rain_processed = molar_conc_seawater(rain)
    
    ## Change the first column, first row name to 'unique_code'
    rain_processed.columns = ['unique_code'] + [col for col in rain_processed.columns[1:]]
    
    # Charge balance
    rain_processed = charge_balance(rain_processed)

    #print(df_neat)
    
    # Initialize lists to store unique codes and NICB values 
    NICB_Balance = NICB_Valid(rain_processed)
    
    #print(NICB_Balance)

    # Ensure columns are in correct order
    rain_processed = rain_processed[['K [aq] (mM)', 'Na [aq] (mM)', 'Cl [aq] (mM)']]
    
    # Filter df to include only valid samples
    df = df[df['unique_code'].isin(df2['unique_code_valid'])]
    
    # Remove row with string "T1BWF-1121" in the unique_code column
    df = df[~df['unique_code'].str.contains('T1BWF-1121', na=False)]

    # Create a copy of df for modifications
    df_copy = df.copy()
    

    #cl_minimum = the lowest cl value in df_copy:
    cl_minimum = df_copy['Cl [aq] (mM)'].min()
    

    # Correction calculation
    

    cl_rain_value = rain_processed['Cl [aq] (mM)'].values[0]
    
    na_rain_value = rain_processed['Na [aq] (mM)'].values[0]
    
    k_rain_value = rain_processed['K [aq] (mM)'].values[0]
    
    cl_corrected_rain_value = cl_rain_value*(na_rain_value/(na_rain_value + k_rain_value))
    
    min_cl = df_copy['Cl [aq] (mM)'].min()
    
    

    if cl_rain_value != 0:

        #df_copy['*Na [aq] (mM)'] = df_copy['Na [aq] (mM)'] - ((na_rain_value/cl_corrected_rain_value) * (df_copy['Cl [aq] (mM)']))
        
        df_copy['*Na [aq] (mM)'] = df_copy['Na [aq] (mM)'] - cl_rain_value
        
        

    # Select relevant columns for final output
    #df_copy = df_copy[['unique_code'] + [col for col in df_copy.columns if '*' in col] + ['latitude_converted', 'longitude_converted', 'NICB', 'calculated_discharge_in_m3_s-1', 'Alkalinity_CaCO3_in_mg_kg-1', 'altitude_in_m', 'water_body']]

    return df_copy















def rotate_points_180(df_lons, df_lats, center_lon, center_lat):
    # Rotate points by 180 degrees around the center
    rotated_lons = 2 * center_lon - df_lons
    rotated_lats = 2 * center_lat - df_lats
    return rotated_lons, rotated_lats




def Xsil_uncorrected(df_neat, NICB_Balance, df_sed, processed_mean_mg_na, processed_sigma_mg_na, processed_mean_ca_na, processed_sigma_ca_na, processed_mean_k_na, processed_sigma_k_na):
    
    df = Chloride_XSil_Correction(df_neat, NICB_Balance)
    
    df_copy = df.copy()
    
    df_copy = df_copy[~df_copy['unique_code'].str.contains('T1DW-0322', na=False)]
    df_copy = df_copy[~df_copy['unique_code'].str.contains('RC15WF-1121', na=False)]
    
    ## Filter df_copy for all those samples to be below -12.35 latitude [i.e. can be -12.50] for it to be defined a silicate lithology
    df_copy = df_copy[df_copy['latitude_converted'] < -12.35]
    
    ## Filter df_sed for all those samples to be below -12.35 latitude [i.e. can be -12.50] for it to be defined a silicate lithology
    df_sed = df_sed[df_sed['latitude_converted'] < -12.35]
    
    CaNa_sil = processed_mean_ca_na
    Ca_Na_sil_std = processed_sigma_ca_na
    
    MgNa_sil = processed_mean_mg_na
    Mg_Na_sil_std = processed_sigma_mg_na
    
    KNa_sil = processed_mean_k_na
    K_Na_sil_std = processed_sigma_k_na
    
    ## Define *Na [aq] (mM):
    # *Na = Na corrected for by the Cl amount of rain that Na is supposed to receive (Na/Na+K)
    
    df_copy['Ca_Sil'] = df_copy['*Na [aq] (mM)'] * CaNa_sil
    df_copy['Mg_Sil'] = df_copy['*Na [aq] (mM)'] * MgNa_sil
    df_copy['K_Sil'] = df_copy['*Na [aq] (mM)'] * KNa_sil
    
    df_copy['X_Sil'] = ((2 * df_copy['Ca_Sil']) + (2 * df_copy['Mg_Sil']) + df_copy['K_Sil'] + df_copy['*Na [aq] (mM)']) / (
                        (2 * df_copy['Ca [aq] (mM)']) + (2 * df_copy['Mg [aq] (mM)']) + df_copy['K [aq] (mM)'] + df_copy['Na [aq] (mM)'])
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df_copy['X_Sil'], df_copy['altitude_in_m'], alpha=0.7, s=70)
    plt.xlabel('XSil')
    plt.ylabel('Altitude')
    plt.title(f'Scatter plot of Altitude vs. XSil Uncorrected - Refugio V. Rainwater')

    # Annotate each point with unique_code
    for index, row in df_copy.iterrows():
        plt.text(row['X_Sil'], row['altitude_in_m'], row['unique_code'], fontsize=8, ha='center', va='bottom')

    plt.savefig('chemweathering/figures/XSil_Uncorrected_Refugio_V_Rainwater.png')
    plt.close()
    
    
    
    
    # Match up sed samples to water samples by closest location to see whether there is any systematic change
    
    
    df_sed_matched = df_sed.copy()
    df_sed_matched['closest_water_sample'] = ''

    for index, row in df_sed_matched.iterrows():
        sed_point = Point(row['longitude_converted'], row['latitude_converted'])
        min_distance = float('inf')
        closest_water_sample = ''

        for _, water_row in df_copy.iterrows():
            water_point = Point(water_row['longitude_converted'], water_row['latitude_converted'])
            distance = sed_point.distance(water_point)

            if distance < min_distance:
                min_distance = distance
                closest_water_sample = water_row['unique_code']

        df_sed_matched.at[index, 'closest_water_sample'] = closest_water_sample

    # Rename columns to avoid conflicts
    df_sed_matched = df_sed_matched.rename(columns={'unique_code': 'sed_unique_code', 'closest_water_sample': 'unique_code'})

    # Merge df_copy and df_sed_matched on unique_code to ensure proper alignment
    df_matched = pd.merge(df_copy, df_sed_matched, on='unique_code')

    # Debugging: Print lengths of the matched DataFrames
    print("Length of df_matched:", len(df_matched))
    print("Length of df_sed_matched:", len(df_sed_matched))
    
    # Plot Ca/Na of sed samples against XSil of water samples
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df_matched['X_Sil'], df_matched['Ca/Na'], alpha=0.7, s=70)
    plt.xlabel('XSil')
    plt.ylabel('Ca/Na of Sediments')
    plt.title('Scatter plot of Ca/Na of Sediments vs. XSil of Water Samples')

    # Annotate each point with unique_code
    for index, row in df_matched.iterrows():
        plt.text(row['X_Sil'], row['Ca/Na'], row['unique_code'], fontsize=8, ha='center', va='bottom')

    plt.savefig('chemweathering/figures/CaNa_vs_XSil.png')
    plt.show()
    plt.close()
    
    
    
    
    
    return(df_copy)


    