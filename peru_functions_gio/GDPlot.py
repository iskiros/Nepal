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



def convert_latitude(row):
    if pd.notna(row['latitude_new']):
        return row['latitude_new']
    elif pd.notna(row['latitude_old']):
        match = re.match(r'(\d+)\s+(\d+\.\d+)', str(row['latitude_old']))
        if match:
            degrees = int(match.group(1))
            minutes = float(match.group(2))
            decimal_degrees = -(degrees + (minutes / 60))
            return decimal_degrees
    return None


def convert_longitude(row):
    # Function to convert degrees and minutes to decimal degrees for longitude

    if pd.notna(row['longitude_new']):
        return row['longitude_new']
    elif pd.notna(row['longitude_old']):
        match = re.match(r'(\d+)\s+(\d+\.\d+)', str(row['longitude_old']))
        if match:
            degrees = int(match.group(1))
            minutes = float(match.group(2))
            decimal_degrees = -(degrees + (minutes / 60))
            return decimal_degrees
    return None



#- Recreates Gaillardet et al, 1999 plot for our samples
#- has been tweaked so it no longer requires running no matter what [issue with not using df_copy]




def GDPlot(df, df2):

    # want to include only NICB valid samples. Takes only the unique code rows from dfneat if they also appear on unique_codes_valid
    
    df_copy = df.copy()
    
    ##################################################################
    
    df_copy = df_copy[df_copy['unique_code'].isin(df2['unique_code_valid'])].copy()
    
    #### Might I just be able to do df instead of dfnew??
    #print(df.columns)

    df_copy.loc[:, 'latitude_converted'] = df_copy.apply(convert_latitude, axis=1)
    df_copy.loc[:, 'longitude_converted'] = df_copy.apply(convert_longitude, axis=1)

    # Drop rows with NaN values in converted latitude and longitude
    df_copy.dropna(subset=['longitude_converted', 'latitude_converted'], inplace=True)



    ##################################################################


    plt.figure(figsize=(10,6))

    df_copy.loc[:, 'HCO3Na'] = df_copy['HCO3 [aq] (mM)']/df_copy['Na [aq] (mM)']
    df_copy.loc[:, 'CaNa'] = df_copy['Ca [aq] (mM)']/df_copy['Na [aq] (mM)']

    plt.scatter(df_copy['HCO3 [aq] (mM)'], df_copy['SO4 [aq] (mM)'], alpha=0.7, s=70, c=df_copy['longitude_converted'], cmap='viridis')
    plt.xlabel('HCO3 [aq] (mM)')
    plt.ylabel('SO4 [aq] (mM)')
    plt.axline((0, 0), (1, 1), linewidth=1, color='b', label='1 HCO3 to 1 SO4')
    plt.axline((0, 0), (2, 1), linewidth=1, color='r', label='2 HCO3 to 1 SO4')
    plt.colorbar(label='Longitude') 
    plt.legend()
    #convert axes to Logarithmic scale
    #plt.xscale("log")
    #plt.yscale("log")

    plt.title('HCO3/SO4 Plot')
    plt.savefig('chemweathering/figures/HCO3SO4.png')
    plt.close()
    #plt.show()