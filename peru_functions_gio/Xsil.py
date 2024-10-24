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





def rotate_points_180(df_lons, df_lats, center_lon, center_lat):
    # Rotate points by 180 degrees around the center
    rotated_lons = 2 * center_lon - df_lons
    rotated_lats = 2 * center_lat - df_lats
    return rotated_lons, rotated_lats




def Xsil(df, processed_mean_mg_na, processed_sigma_mg_na, processed_mean_ca_na, processed_sigma_ca_na):
    
    df_copy = df.copy()
    
    df_copy = df_copy[~df_copy['unique_code'].str.contains('T1DW-0322', na=False)]
    df_copy = df_copy[~df_copy['unique_code'].str.contains('RC15WF-1121', na=False)]
    
    #print(df_copy)
    
    CaNa_sil = processed_mean_ca_na
    
    #CaNa_sil = 0.59
    
    Ca_Na_sil_std = processed_sigma_ca_na
    
    MgNa_sil = processed_mean_mg_na
    
    #MgNa_sil = 0.36
    
    Mg_Na_sil_std = processed_sigma_mg_na
    #Galy and France Lanord say it is 0.2 on avg, we have gotten it primarily from bulk rock data, with a bit of log normal tweaking
    
    
    df_copy['Ca_Sil'] = df_copy['+Na [aq] (mM)'] * CaNa_sil
    
    df_copy['Mg_Sil'] = df_copy['+Na [aq] (mM)'] * MgNa_sil
    
    df_copy['X_Sil'] = ((2*df_copy['Ca_Sil']) + (2*df_copy['Mg_Sil']) + df_copy['+K [aq] (mM)'] + df_copy['+Na [aq] (mM)'])/((2*df_copy['+Ca [aq] (mM)']) + (2*df_copy['+Mg [aq] (mM)']) + df_copy['+K [aq] (mM)'] + df_copy['+Na [aq] (mM)'])
    
    #df_copy['X_Sil'] = ((df_copy['Ca_Sil']/2) + (df_copy['Mg_Sil']/2) + df_copy['*K [aq] (mM)'] + df_copy['*Na [aq] (mM)'])/((2*df_copy['*Ca [aq] (mM)']) + (2*df_copy['*Mg [aq] (mM)']) + df_copy['*K [aq] (mM)'] + df_copy['*Na [aq] (mM)'])
    
    # Filter out X_Sil > 1.0 Samples, and print them "these samples were >1"
    filtered_df = df_copy[df_copy['X_Sil'] > 1.0]
    print("These samples had XSil >1:")
    for index, row in filtered_df.iterrows():
        print(row['unique_code'])
    
    
    #### Changed so it forces to 1.0
    
    # Set any value greater than 1 to 1.0
    df_copy.loc[df_copy['X_Sil'] > 1.0, 'X_Sil'] = 1.0
    
    
    plt.figure(figsize=(10,6))
    scatter = plt.scatter(df_copy['X_Sil'], df_copy['altitude_in_m'], alpha=0.7, s=70)
    plt.xlabel('XSil')
    plt.ylabel('Altitude')
    
    plt.title(f'Scatter plot of Altitude vs. XSil - Refugio V. Rainwater - Corrected for evaporites')

            # Annotate each point with unique_code
    for index, row in df_copy.iterrows():
        plt.text(row['X_Sil'], row['altitude_in_m'], row['unique_code'], fontsize=8, ha='center', va='bottom')

    #plt.legend()  # Include legend with labels
    plt.savefig('chemweathering/figures/XSil_Refugio_V_Rainwater.png') 
    #plt.show()  # Show each plot individually #
    plt.close()
    return(df_copy)
