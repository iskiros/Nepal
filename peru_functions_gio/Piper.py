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

# - Using plotly to make a piper plot for our samples
# - Can be run using corrected OR uncorrected Data
# - Keep in mind if you are plotting discharge it would be good to plot it after the fraction calculation 
# as it will calculate discharges for tributaries which don’t have an immediate value


def Piper(df):
    
    df_copy = df.copy()
    
    #print(df_copy)
    
    
    #we are doing uncorrected for now
    
    
    
    
    ########## DATA FRAME MAINTENANCE ##########
    
    # Drop rows with NaN values in discharge
    df_copy.dropna(subset=['calculated_discharge_in_m3_s-1'], inplace=True)
    
    
    # Ensure all values in 'calculated_discharge_in_m3_s-1' are numeric
    df_copy['calculated_discharge_in_m3_s-1'] = pd.to_numeric(df_copy['calculated_discharge_in_m3_s-1'], errors='coerce')
    
    # Drop any rows that could not be converted to numeric values
    df_copy.dropna(subset=['calculated_discharge_in_m3_s-1'], inplace=True)
    
    
    ####### now do the same but for alkalinity
    
    # Drop rows with NaN values in alkalinity
    df_copy.dropna(subset=['Alkalinity_CaCO3_in_mg_kg-1'], inplace=True)
    
    
    # Ensure all values in alkalinity are numeric
    df_copy['Alkalinity_CaCO3_in_mg_kg-1'] = pd.to_numeric(df_copy['Alkalinity_CaCO3_in_mg_kg-1'], errors='coerce')
    
    # Drop any rows that could not be converted to numeric values
    df_copy.dropna(subset=['Alkalinity_CaCO3_in_mg_kg-1'], inplace=True)

    
    # Convert the size column to a list or else it gives an invalid error
    #size_list = df_copy['calculated_discharge_in_m3_s-1'].tolist()
    
    #those values are too small to work with
    



    ########## LOG TRANSFORMATION ##########
    
    # Apply a logarithmic transformation to the size column
    log_size = np.log(df_copy['calculated_discharge_in_m3_s-1'] + 1)
    
    #########################################
    

    
    #########################################
    
    df_copy['Na+K'] = df_copy['Na [aq] (mM)'] + df_copy['K [aq] (mM)']
    
    fig1 = px.scatter_ternary(df_copy, a="Mg [aq] (mM)", b="Ca [aq] (mM)", c="Na+K", hover_name="unique_code", color="water_body", size='altitude_in_m', size_max=15,
                             color_discrete_map = {"mainstream": "blue", "tributary": "green", "spring":"red"})

    #fig1.show()
    
    #########################################
    
    
    
    #########################################
    
    
    df_copy['Na+K (mM)'] = df_copy['Na [aq] (mM)'] + df_copy['K [aq] (mM)']
    
    df_copy['CO3+HCO3 (mM)'] = df_copy['Alkalinity_CaCO3_in_mg_kg-1'] / 100.0869
    
    fig2 = px.scatter_ternary(df_copy, a="CO3+HCO3 (mM)", b="Cl [aq] (mM)", c="SO4 [aq] (mM)", hover_name="unique_code", color="water_body", size='altitude_in_m', size_max=15,
                             color_discrete_map = {"mainstream": "blue", "tributary": "green", "spring":"red"})

    #fig2.show()
    
    #########################################
    
    
    
    #########################################
    
    ### Now plotting SO4 and Na+K
    
    df_copy['Na+K Normal (mM)'] = df_copy['Na+K (mM)'] / (df_copy['K [aq] (mM)'] + df_copy['Na [aq] (mM)'] + df_copy['Ca [aq] (mM)'] + df_copy['Mg [aq] (mM)'])
    
    df_copy['SO4 Normal (mM)'] = df_copy['SO4 [aq] (mM)'] / (df_copy['CO3+HCO3 (mM)'] + df_copy['SO4 [aq] (mM)'] + df_copy['Cl [aq] (mM)'])
    
    plt.figure(figsize=(10,6))

    plt.scatter(df_copy['Na+K Normal (mM)'], df_copy['SO4 Normal (mM)'], alpha=0.7, s=70, c=df_copy['altitude_in_m'], cmap='viridis')
    plt.xlabel('Na+K Normal (mM)')
    plt.ylabel('SO4 Normal (mM)')
    #plt.axline((0, 0), (1, 1), linewidth=1, color='b', label='1 HCO3 to 1 SO4')
    #plt.axline((0, 0), (2, 1), linewidth=1, color='r', label='2 HCO3 to 1 SO4')
    plt.colorbar(label='Altitude (m)') 
    #plt.legend()
    #convert axes to Logarithmic scale
    #plt.xscale("log")
    #plt.yscale("log")

    plt.title('Normalised Na+K and SO4 (Molar)')
    plt.savefig('chemweathering/figures/Molar-Prop.png')
    #plt.show()
    plt.close()
    
    #########################################
    
    
    return(df_copy)
    
