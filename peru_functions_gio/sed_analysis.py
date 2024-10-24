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
# - Separate to main water investigation
# - Used to calculate Ca/Na and Mg/Na ratios from our sediments, 
# under the assumption that the silicate rocks will give lower values than the carbonate rocks 
# so if you take the modal estiamte of the sililcate derived region you will get Ca/Na and Mg/Na ratios
# - Not very quantitative though, can be done by eye to give our modal Ca/Na and Mg/Na 
# ratios for our silicate seds (0.59 and 0.36 respectively)




def slice_relevant_datas(df):
    """Slice relevant data from the dataset"""
    # select major elements for charge balance
    elements = {'Ca', 'Mg', 'K', 'Na', 'HCO3', 'Cl', 'SO4', 'Al', 'As', 'Ba', 'Bi', 'Ce', 'Fe', 'Li', 'Rb', 'Sr', 'CO3', 'H2PO4'}

    # loop through columns to retrieve columns with major elements in name with element+_in_mg_kg-1
    # save as a list
    major_elements = []
    for element in elements:
        col_standard = f"{element}_in_mg_kg-1"
        col_alternate = f"{element}_in_mg_kg-1 [MA-0722]"
        if col_standard in df.columns:
            major_elements.append(col_standard)
        elif col_alternate in df.columns:
            major_elements.append(col_alternate)



    # Add more columns from the original dataframe
    additional_columns = ['longitude_new', 'longitude_old', 'latitude_new', 'latitude_old', 'calculated_discharge_in_m3_s-1', 'Alkalinity_CaCO3_in_mg_kg-1', 'altitude_in_m', 'water_body']  # Add more column names as needed
    major_elements += additional_columns

    # slice the dataframe to include only the major elements
    df_slice = df[major_elements]
    


    # append the sample ID to the sliced dataframe
    df_slice.insert(0, 'unique_code', df['unique_code'])
    
    #The first argument, 0, specifies the index position where the new column should be inserted. 
    # In this case, it is inserted at the beginning of the DataFrame.

    #The second argument, 'unique_code', is the name of the new column that will be inserted.

    #The third argument, df['unique_code'], is the data that will be populated in the new column. 
    # It is taken from the 'unique_code' column of the df DataFrame.
    
    

    #print(df_slice.columns)
    
    #print(df_slice)

    return df_slice


def convert_latitudes(row):
    if pd.notna(row['latitude_new']):
        return row['latitude_new']
    elif pd.notna(row['latitude_old']):
        match = re.match(r'(\d+)\s+(\d+\.\d+)', str(row['latitude_old']))
        if match:
            degrees = int(match.group(1))
            minutes = float(match.group(2))
            decimal_degrees = degrees + (minutes / 60)
            return decimal_degrees
    return None


def convert_longitudes(row):
    if pd.notna(row['longitude_new']):
        return row['longitude_new']
    elif pd.notna(row['longitude_old']):
        match = re.match(r'(\d+)\s+(\d+\.\d+)', str(row['longitude_old']))
        if match:
            degrees = int(match.group(1))
            minutes = float(match.group(2))
            decimal_degrees = degrees + (minutes / 60)
            return decimal_degrees
    return None


def calculate_molar_concentrations(df):
    """Calculate molar concentration of the major elements"""
    element_dict = {
        'molecules': ['Ca', 'Mg', 'K', 'Na', 'HCO3', 'Cl', 'SO4', 'Al', 'As', 'Ba', 'Bi', 'Ce', 'Fe', 'Li', 'Rb', 'Sr', 'CO3', 'H2PO4'],
        'molar_mass': [40.08, 24.31, 39.10, 22.99, 61.02, 35.45, 96.06, 26.98, 74.92, 137.33, 208.98, 140.12, 55.85, 6.94, 85.47, 87.62, 60.01, 97.99],
        'valency': [2, 2, 1, 1, 1, 1, 2, 3, 3, 2, 3, 3, 2, 1, 1, 2, 2, 1]
    }

    # Suffixes
    suffix = '_in_mg_kg-1'
    alt_suffix = '_in_mg_kg-1 [MA-0722]'

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
        df_copy.loc[:, element + ' (mM)'] = df_copy[col_name] / molar_mass
        df_copy.loc[:, element + ' (meq/L)'] = df_copy.loc[:, element + ' (mM)'] * valency

    #print(df_copy.columns)

    # Select relevant columns
    df_copy = df_copy[['unique_code'] + [col for col in df_copy.columns if ' (mM)' in col] + [col for col in df_copy.columns if ' (meq/L)' in col]+ ['latitude_converted'] + ['longitude_converted'] + ['altitude_in_m']]

    #print(df_copy)

    return df_copy
    


def analyze_mgca(df):
    #print(df_neat)
    
    
    ###################### DATA CLEANING ######################
    
    
    df_copy = df.copy()
    
    df_copy['Ca/Na'] = df_copy['Ca (mM)']/df_copy['Na (mM)']
    
    # Convert Ca/Na to numeric, coercing errors to NaN
    df_copy['Ca/Na'] = pd.to_numeric(df_copy['Ca/Na'], errors='coerce')

    
    df_copy = df_copy[(df_copy['Ca/Na'] > 0) & np.isfinite(df_copy['Ca/Na'])]
    

    # remove NaN values:
    df_copy = df_copy.dropna(subset=['Ca/Na'])
    
    
    #######################################################
    
    
    
    
    
    #print(df_copy['Ca/Na'])
    #print how many rows there are in Ca/Na
    #print(df_copy['Ca/Na'].count())
    
    # what is the unique code of the lowest 10 Ca/Na values, with the corresponding Ca/Na values
    #print(df_copy.nsmallest(10, 'Ca/Na')[['unique_code', 'Ca/Na']])
    
    
    ###################### PLOTTING ######################
    
    # Log normalise Ca/Na then plot a histogram:
    df_copy['Ca/Na_log'] = np.log(df_copy['Ca/Na'])
    
    #Plot a histogram of Ca/Na against density:
    plt.figure(figsize=(10,6))
    sns.histplot(df_copy['Ca/Na'], kde=True, bins=5000)
    plt.xlabel('(Ca/Na)')
    plt.ylabel('Density')
    plt.title('Histogram of (Ca/Na) values')
    #plt.show()
    plt.close()
    
    
    #############################################################################
    
    
    
    df_copy['Mg/Na'] = df_copy['Mg (mM)'] / df_copy['Na (mM)']
    
    # Convert Mg/Na to numeric, coercing errors to NaN
    df_copy['Mg/Na'] = pd.to_numeric(df_copy['Mg/Na'], errors='coerce')

    df_copy = df_copy[(df_copy['Mg/Na'] > 0) & np.isfinite(df_copy['Mg/Na'])]
    
    # remove NaN values:
    df_copy = df_copy.dropna(subset=['Mg/Na'])
    
    
    
    #############################################################################
    
    
    
    
    #print(df_copy['Mg/Na'])
    #print how many rows there are in Mg/Na
    
    
    # what is the unique code of the lowest 10 Mg/Na values, with the corresponding Mg/Na values
    #print(df_copy.nsmallest(10, 'Mg/Na')[['unique_code', 'Mg/Na']])
    
    
    
    ###################### PLOTTING ######################
    
    # Log normalise Mg/Na then plot a histogram:
    df_copy['Mg/Na_log'] = np.log(df_copy['Mg/Na'])
    
    #Plot a histogram of Mg/Na against density:
    plt.figure(figsize=(10,6))
    sns.histplot(df_copy['Mg/Na'], kde=True, bins=5000)
    plt.xlabel('(Mg/Na)')
    plt.ylabel('Density')
    plt.title('Histogram of (Mg/Na) values')
    #plt.show()
    plt.close()
    
    #######################################################
    
    
    # What is the mode?
    #mode_value = df_copy['Ca/Na'].mode()[0]

    #print("Mode of Ca/Na:", mode_value)
        
    
    #print(df_copy['Ca/Na'].min())
    
    #print(df_copy['Ca/Na'].max())
    
    #######################################################
    
    
    
    ##################################################################
    
    df_copy['K/Na'] = df_copy['K (mM)'] / df_copy['Na (mM)']

    # Convert K/Na to numeric, coercing errors to NaN
    df_copy['K/Na'] = pd.to_numeric(df_copy['K/Na'], errors='coerce')

    df_copy = df_copy[(df_copy['K/Na'] > 0) & np.isfinite(df_copy['K/Na'])]

    # remove NaN values:
    df_copy = df_copy.dropna(subset=['K/Na'])
    
    ##################################################################
    
    
    

    #print(df_copy['K/Na'])
    #print how many rows there are in K/Na

    # what is the unique code of the lowest 10 K/Na values, with the corresponding K/Na values
    #print(df_copy.nsmallest(10, 'K/Na')[['unique_code', 'K/Na']])
    
    
    
    ##################################################################

    # Log normalise K/Na then plot a histogram:
    df_copy['K/Na_log'] = np.log(df_copy['K/Na'])

    #Plot a histogram of K/Na against density:
    plt.figure(figsize=(10,6))
    sns.histplot(df_copy['K/Na'], kde=True, bins=5000)
    plt.xlabel('(K/Na)')
    plt.ylabel('Density')
    plt.title('Histogram of (K/Na) values')
    #plt.show()
    plt.close()
    
    ##################################################################
    
    return df_copy



def sed_analysis():

    ###################### CONTROLLING SCRIPT ######################
    df = pd.read_excel('chemweathering/data/Canete_Long_Data_Revisited_Seds.xlsx', sheet_name='Data')
    
    # Slice waters
    df = df[(df['sample_type'] == 'sediment') | (df['sample_type'] == 'rock')]

    # Convert 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Extract month and year from 'date' column
    df['month_year'] = df['date'].dt.strftime('%m%y')
    
    # For samples after row 290, change 'unique_code' to a concatenation of 'site' and 'month_year'
    df.loc[344:, 'unique_code'] = df.loc[340:, 'site'] + '-' + df.loc[340:, 'month_year']
    
    #print(df['unique_code'])
    

    # Slice relevant data
    df_slice = slice_relevant_datas(df)

    
        #### Might I just be able to do df instead of dfnew??
    #print(df.columns)

    df_slice['latitude_converted'] = df_slice.apply(convert_latitudes, axis=1)
    df_slice['longitude_converted'] = df_slice.apply(convert_longitudes, axis=1)
  
    
    

    # Drop rows with NaN values in converted latitude and longitude
    df_slice.dropna(subset=['longitude_converted', 'latitude_converted'], inplace=True)
    

    
    # Make sure longitude_converted and latitude_converted are all negative [absolute value then negative]
    df_slice['longitude_converted'] = -abs(df_slice['longitude_converted'])
    df_slice['latitude_converted'] = -abs(df_slice['latitude_converted'])
    
    

    # Replace non-numeric values with NaN in the dataframe except the first column and the last columns, which are unique_code and water_body
    df_slice.loc[:, df_slice.columns[1:-1]] = df_slice.loc[:, df_slice.columns[1:-1]].apply(pd.to_numeric, errors='coerce')
    

    # Drop rows with NaN values in 'unique_code' column
    df_slice = df_slice.dropna(subset=['unique_code'])
    
    
    ## Drop columns that contain no numerical values:
    df_slice = df_slice.dropna(axis=1, how='all')
    
    

    # Create molar columns
    df_neat = calculate_molar_concentrations(df_slice)
    
    
    df_mgca = analyze_mgca(df_neat)
    

    
    #print(df_mgca.columns)
    

    return df_mgca
    
    
   
