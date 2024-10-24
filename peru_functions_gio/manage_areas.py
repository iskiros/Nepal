import sys
import os
import math
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




def manage_areas(df_fraction):
    

    # Calculate cumulative area downstream of each site.
    # Rules:
    
    # 1. Mainstream Filtering and Processing:

    # • Only samples with unique_code starting with "R" (indicating mainstream sites) are considered for mainstream processing.

    # • The unique_code for mainstream sites is trimmed to the first 4 characters to extract the relevant part (e.g., "RC07").

    # • The numeric part of the unique_code (e.g., "07" from "RC07") is extracted for sorting and comparison.



    # 2. Tributary Filtering and Processing:
    # • Only samples with unique_code starting with "T" (indicating tributary sites) are considered for tributary processing.

    # • The unique_code for tributary sites is trimmed to the first 3 characters to extract the relevant part (e.g., "T7a").

    # • The numeric part of the unique_code (e.g., "7" from "T7a") is extracted and used to match tributaries with the corresponding mainstream site.

    # • If there are multiple tributaries with the same numeric code (e.g., multiple "T7" tributaries), the maximum area among them is used.



    # 3. Sorting:
    # • The mainstream sites are sorted in descending order based on the numeric part of their unique_code. This ensures that the cumulative area is calculated from upstream to downstream (e.g., RC16, RC15, etc.).

    # • Tributaries are also processed based on their numeric code, matching them to their corresponding mainstream site.

    # 4. Cumulative Area Calculation:
    # • The calculation begins with the most upstream mainstream site (e.g., RC16).

    # • For each mainstream site:
    #     ◦ If there are matching tributaries (e.g., T7 for RC07), the maximum area from these tributaries is added to the cumulative area for that mainstream site.
    #     ◦ The area of the mainstream site itself is then added to this cumulative area.
    #     ◦ The cumulative area is then stored for the current mainstream site.

    # • The cumulative area for each mainstream site includes the area from all downstream sites and their associated tributaries.




    # 5. Preservation of Tributary Areas:
    # • Each tributary retains its original maximum area in the final DataFrame, but its contribution is added to the cumulative area of the corresponding mainstream site.
    # • One thing we are ignoring is that the tributaries themselves have branches, but we are just considering the mainstream for now

    # 6. Final Assignment:
    # The calculated cumulative_area for both mainstream and tributary sites is assigned back to the original DataFrame (df_copy), ensuring that all areas are correctly represented.

    # Filter the dataframe to include only mainstream samples
    # convert df_copy['unique_code'] to string
    
    
    
    
    
    ############################################################################################################
    
    # Load the CSV file
    df = pd.read_csv('/Users/enrico/Desktop/ROKOS Internship/QGIS/area_df.csv')

    # Swap the first and second columns
    df = df[['total_area_m2', 'unique_code', 'geometry']]

    # Resetting the index to move the current index into a column
    df_fixed = df.reset_index()

    # Rename the columns correctly
    df_fixed.columns = ['unique_code', 'geometry', 'total_area_m2', 'geometry2']

    # Display the fixed DataFrame
    #print(df_fixed.head())
    
    # Keep only the first two columns:
    #df_fixed = df_fixed.iloc[:, :2]
    
    df_copy = df_fixed.copy()
    
    # Ensure the unique_code column is properly named
    df_copy.rename(columns={df_copy.columns[0]: 'unique_code'}, inplace=True)
    
    # Ensure the df_fraction unique_code column is properly named
    df_fraction.rename(columns={df_fraction.columns[0]: 'unique_code'}, inplace=True)
    
    # Calculate cumulative area downstream of each site
    df_copy['unique_code'] = df_copy['unique_code'].astype(str)
    
  
    
    #print(df_copy['total_area_m2'])
    
    ############################################################################################################
    
    # Filter the dataframe to include only mainstream samples and explicitly make a copy
    df_mainstream = df_copy[df_copy['unique_code'].str.startswith('R')].copy()
    df_mainstream['unique_code_trimmed'] = df_mainstream['unique_code'].str[:4]

    #print(df_mainstream['unique_code_trimmed'])

    # Filter the dataframe to include only tributary samples and explicitly make a copy
    df_trib = df_copy[df_copy['unique_code'].str.startswith('T')].copy()
    df_trib['unique_code_trimmed'] = df_trib['unique_code'].str[:3]
    df_trib['unique_code_trimmed'] = df_trib['unique_code_trimmed'].str.extract('T(\d+)')[0].astype(int)  # Extract the numeric part



    ############################################################################################################

    # Sort the mainstream dataframe by the numeric part of unique_code in descending order
    df_mainstream['numeric_code'] = df_mainstream['unique_code_trimmed'].str.extract('(\d+)').astype(int)
    df_mainstream.sort_values('numeric_code', ascending=False, inplace=True)



    # Initialize a new column 'cumulative_area' with NaN values
    df_mainstream['cumulative_area'] = np.nan
    df_trib['cumulative_area'] = np.nan  # Initialize the cumulative_area column for tributaries


    ############################################################################################################


    # Iterate over the tributary dataframe to assign the maximum area per unique_code_trimmed
    for numeric_code in df_trib['unique_code_trimmed'].unique():
        max_area = df_trib[df_trib['unique_code_trimmed'] == numeric_code]['total_area_m2'].max()
        df_trib.loc[df_trib['unique_code_trimmed'] == numeric_code, 'cumulative_area'] = max_area

    # Initialize a variable to keep track of the cumulative area
    cumulative_area = 0


    ############################################################################################################

    # Iterate over the mainstream rows in descending order of numeric code
    for index, row in df_mainstream.iterrows():
        unique_code_trimmed = row['unique_code_trimmed']
        numeric_code = row['numeric_code']

        # Find all rows with the same unique_code_trimmed in mainstream
        same_site_rows = df_mainstream[df_mainstream['unique_code_trimmed'] == unique_code_trimmed]

        # Add the area of the corresponding tributary (if exists) to the cumulative area
        if numeric_code in df_trib['unique_code_trimmed'].values:
            cumulative_area += df_trib.loc[df_trib['unique_code_trimmed'] == numeric_code, 'cumulative_area'].max()

        # Get the maximum area for the current mainstream site
        max_area = same_site_rows['total_area_m2'].max()

        # Add the maximum area to the cumulative area
        cumulative_area += max_area

        # Assign the cumulative area to all rows with the same unique_code_trimmed in mainstream
        df_mainstream.loc[df_mainstream['unique_code_trimmed'] == unique_code_trimmed, 'cumulative_area'] = cumulative_area


    ############################################################################################################

    # Drop the helper columns used for sorting
    df_mainstream.drop(columns=['numeric_code', 'unique_code_trimmed'], inplace=True)
    df_trib.drop(columns=['unique_code_trimmed'], inplace=True)

    # Update the original dataframe with the cumulative areas for both mainstream and tributaries
    df_copy.loc[df_copy['unique_code'].str.startswith('R'), 'cumulative_area'] = df_mainstream['cumulative_area']
    df_copy.loc[df_copy['unique_code'].str.startswith('T'), 'cumulative_area'] = df_trib['cumulative_area']
    
    #print(df_copy['cumulative_area'])
    
    
    ############################################################################################################
    
    
    ### Add the cumulative area to the fraction dataframe
    
    # Merge df_fraction with the relevant columns from df_copy based on unique_code
    df_fraction = df_fraction.merge(df_copy[['unique_code', 'cumulative_area']], on='unique_code', how='left')
    
    # Export df_fraction to a excel file
    df_fraction.to_excel('/Users/enrico/Desktop/ROKOS Internship/QGIS/fraction_df_areas.xlsx', index=False)
    
    
    ############################################################################################################
    
    return df_fraction
            