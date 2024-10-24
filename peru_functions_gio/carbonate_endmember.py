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




######### AFTER CORRECTION, CALCULATING MOLAR RATIOS IN AN ATTEMPT TO FIGURE OUT CARBONATE ENDMEMBERS #########

######### AFTER BICKLE ET AL, 2005 ##########


def carbonate_endmember(df):
    
    
    ######### DATAFRAME MANIPULATION #########
    
    ## Plot Cation/Ca ratios against Na/Ca ratios
    # Define the figure
    
    df_copy = df.copy()
    
    #print(df_copy.columns)
    
    #############################################
    
    
    
    
    ######### CALCULATING MOLAR RATIOS #########
    
    #print(df_copy.columns)
    df_copy['Na/Ca'] = df_copy['+Na [aq] (mM)'] / df_copy['+Ca [aq] (mM)']
    df_copy['Mg/Ca'] = df_copy['+Mg [aq] (mM)'] / df_copy['+Ca [aq] (mM)']
    df_copy['K/Ca'] = df_copy['+K [aq] (mM)'] / df_copy['+Ca [aq] (mM)']
    df_copy['Ca/Na'] = df_copy['+Ca [aq] (mM)'] / df_copy['+Na [aq] (mM)']
    df_copy['Mg/Na'] = df_copy['+Mg [aq] (mM)'] / df_copy['+Na [aq] (mM)']
    
    #############################################
    


    ######### PLOTTING THE FIGURES. 3 in total #########

    
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ##################


    # Plot Mg/Ca vs Na/Ca
    sc = ax[0].scatter(df_copy['Na/Ca'], df_copy['Mg/Ca'], c=df_copy['altitude_in_m'], cmap='viridis', vmin=df_copy['altitude_in_m'].min(), vmax=df_copy['altitude_in_m'].max(), s=50)
    ax[0].set_xlabel('Na/Ca')
    ax[0].set_ylabel('Mg/Ca')
    ax[0].set_title('Mg/Ca vs Na/Ca')
    fig.colorbar(sc, ax=ax[0], label='Altitude (m)')

    # Add sample names
    for i, row in df_copy.iterrows():
        ax[0].text(row['Na/Ca'], row['Mg/Ca'], row['unique_code'], fontsize=8, ha='center', va='center')


    ##################


    # Plot K/Ca vs Na/Ca
    sc = ax[1].scatter(df_copy['Na/Ca'], df_copy['K/Ca'], c=df_copy['altitude_in_m'], cmap='viridis', vmin=df_copy['altitude_in_m'].min(), vmax=df_copy['altitude_in_m'].max(), s=50)
    ax[1].set_xlabel('Na/Ca')
    ax[1].set_ylabel('K/Ca')
    ax[1].set_title('K/Ca vs Na/Ca')
    fig.colorbar(sc, ax=ax[1], label='Altitude (m)')

    # Add sample names
    for i, row in df_copy.iterrows():
        ax[1].text(row['Na/Ca'], row['K/Ca'], row['unique_code'], fontsize=8, ha='center', va='center')
    
    
    ##################
    
    
    # Plot Ca/Na vs Mg/Na
    
    sc = ax[2].scatter(df_copy['Mg/Na'], df_copy['Ca/Na'], c=df_copy['altitude_in_m'], cmap='viridis', vmin=df_copy['altitude_in_m'].min(), vmax=df_copy['altitude_in_m'].max(), s=50)
    ax[2].set_xlabel('Mg/Na')
    ax[2].set_ylabel('Ca/Na')
    ax[2].set_title('Ca/Na vs Mg/Na')
    fig.colorbar(sc, ax=ax[2], label='Altitude (m)')

    # Add sample names
    for i, row in df_copy.iterrows():
        ax[2].text(row['Mg/Na'], row['Ca/Na'], row['unique_code'], fontsize=8, ha='center', va='center')


    ##################
    
    

    plt.tight_layout()
    #plt.show()
    plt.close()
    
    
    
    
    return 