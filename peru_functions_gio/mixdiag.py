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

# Plotting ratios of Mg/Na against Ca/Na and HCO3/Na against Ca/Na
# to see if there is a difference between the water bodies




def mixdiag(df):
    
    df_copy = df.copy()
    
    df_copy = df_copy[~df_copy['unique_code'].str.contains('T1DW-0322', na=False)]
    
    #########
    
    # want to differentiate between River and tributary and spring
    #########
    
    ### print(df_copy.sort_values(by='altitude_in_m', ascending=False))
    ### Just checking highest values

    
   # Ensure all water_body values are mapped to valid markers
    markers = {'mainstream': 'o', 'tributary': '^', 'spring': '*'}

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(10, 6))


    df_copy['MgNa'] = df_copy['*Mg [aq] (mM)'] / df_copy['*Na [aq] (mM)']
    df_copy['CaNa'] = df_copy['*Ca [aq] (mM)'] / df_copy['*Na [aq] (mM)']

    for water_body, marker in markers.items():
        subset = df_copy[df_copy['water_body'] == water_body]
        ax.scatter(subset['*Ca [aq] (mM)'] / subset['*Na [aq] (mM)'],
               subset['*Mg [aq] (mM)'] / subset['*Na [aq] (mM)'],
               s=70, alpha=0.7,
               c=subset['altitude_in_m'],
               cmap='viridis',
               marker=marker)

    ax.set_xlabel('Ca/Na molar ratio')
    ax.set_ylabel('Mg/Na molar ratio')
    ax.set_title('Mg/Na against Ca/Na')
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    # Add colorbar separately
    cb = fig.colorbar(ax.collections[0], ax=ax, label='Altitude (m)')

    # Create legend with custom markers and labels
    for water_body, marker in markers.items():
        ax.scatter([], [], marker=marker, color='black', label=water_body)

    ax.legend(scatterpoints=1, labelspacing=1, title='Water Body', loc='upper left')
    
    #plt.legend()  # Include legend with labels
    plt.savefig('chemweathering/figures/MgNa-CaNa.png')     
    plt.close(fig)
    
    #plt.show()
    
     # Annotate each point with unique_code
    #for index, row in df_copy.iterrows():
    #    plt.text(row['Distance (km)'], row[column], row['unique_code'], fontsize=8, ha='center', va='bottom')
    
    
    ########## NEW PLOT
    
    
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_copy['HCO3Na'] = df_copy['*HCO3 [aq] (mM)'] / df_copy['*Na [aq] (mM)']
    
    for water_body, marker in markers.items():
        subset = df_copy[df_copy['water_body'] == water_body]
        ax.scatter(subset['*Ca [aq] (mM)'] / subset['*Na [aq] (mM)'],
               subset['*HCO3 [aq] (mM)'] / subset['*Na [aq] (mM)'],
               s=70, alpha=0.7,
               c=subset['altitude_in_m'],
               cmap='viridis',
               marker=marker)

    ax.set_xlabel('Ca/Na molar ratio')
    ax.set_ylabel('HCO3/Na molar ratio')
    ax.set_title('HCO3/Na against Ca/Na')
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    # Add colorbar separately
    cb = fig.colorbar(ax.collections[0], ax=ax, label='Altitude (m)')

    # Create legend with custom markers and labels
    for water_body, marker in markers.items():
        ax.scatter([], [], marker=marker, color='black', label=water_body)

    ax.legend(scatterpoints=1, labelspacing=1, title='Water Body', loc='upper left')
    
    plt.savefig('chemweathering/figures/HCO3Na-CaNa.png') 
    plt.close(fig)
    #plt.show()  # Show each plot individually

