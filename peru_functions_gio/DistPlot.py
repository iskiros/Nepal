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


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two sets of lat/lon coordinates.
    """
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(float(lat1))
    lon1_rad = math.radians(float(lon1))
    lat2_rad = math.radians(float(lat2))
    lon2_rad = math.radians(float(lon2))

    # Compute differences in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in kilometers
    distance = R * c
    return distance


# - Calculates the distance to/from shore and plots altitude and all of the corrected elements
# - Not as of yet used after the first two weeks


def DistPlot(df):
    
    ##### Calculate distance from shore #####
    fixed_lat = -13.32468333
    fixed_lon = -76.52861944
    
    ########################################

    # Apply the haversine_distance function to each row in the DataFrame
    df_copy = df.copy()
    df_copy['Distance (km)'] = df_copy.apply(
        lambda row: haversine_distance(fixed_lat, fixed_lon, row['latitude_converted'], row['longitude_converted']),
        axis=1
    )

    ## Nice use of lambda
    
    #print(df_copy)


    ########################################

    max_value_row = df_copy.loc[df_copy['*K [aq] (mM)'].idxmax()]
    #print(df_copy['*K [aq] (mM)'].idxmax())
    #print(max_value_row['unique_code']) 
    
    ## Want to remove row with RC15c-44748
    
    df_copy.to_csv('chemweathering/data/CorrectedValues.csv', index=False)
    
    df_copy = df_copy[~df_copy['unique_code'].str.contains('T1DW-0322', na=False)]
    
    ########################################

    
    for column in df_copy.columns:
        if '*' in column:
            plt.figure(figsize=(10,6))
            scatter = plt.scatter(df_copy['Distance (km)'], df_copy[column], alpha=0.7, s=70, c=df_copy['altitude_in_m'], cmap='viridis', label=column)
            plt.xlabel('Distance (km) from Shore')
            plt.ylabel(column)
            plt.colorbar(label='Altitude (m)')
            plt.title(f'Scatter plot of {column} vs. Distance (km)')

            # Annotate each point with unique_code
            #for index, row in df_copy.iterrows():
            #    plt.text(row['Distance (km)'], row[column], row['unique_code'], fontsize=8, ha='center', va='bottom')

            plt.legend()  # Include legend with labels
            plt.savefig('chemweathering/figures/' + column.replace('*', '').strip() + '.png') 
            #plt.show()  # Show each plot individually
            plt.close()
     