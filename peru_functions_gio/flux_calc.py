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


def flux_calc(df):
    
    df_area = df.copy()
    
        # These are the columns of interest
       # These are the columns of interest
    columns_of_interest = [
        'Carbonate_Carbonic_Short_Term_Mass',
        'Carbonate_Carbonic_Long_Term_Mass',
        'Carbonate_Sulfuric_Short_Term_Mass',
        'Carbonate_Sulfuric_Long_Term_Mass',
        'Silicate_Carbonic_Short_Term_Mass', 
        'Silicate_Carbonic_Long_Term_Mass',
        'Silicate_Sulfuric_Short_Term_Mass', 
        'Silicate_Sulfuric_Long_Term_Mass',
        'cumulative_area'
    ]
    
    # They are in kton CO2/yr. I want to divide them by the area of the site to get the flux in kton CO2/yr/m^2
    
    # Calculate the area of the site
    site_area = df_area['cumulative_area']
    
    # Divide the mass columns by the area to get the flux - in kg/km^2/yr
    
    # times by 10^6 to go from kiloton to kilograms
    # times by 10^6 to go from m^-2 to km^-2
    
    df_area['Carbonate_Carbonic_Short_Term_Flux'] = df_area['Carbonate_Carbonic_Short_Term_Mass'] * 10**12 / site_area
    df_area['Carbonate_Carbonic_Long_Term_Flux'] = df_area['Carbonate_Carbonic_Long_Term_Mass'] * 10**12 / site_area
    df_area['Carbonate_Sulfuric_Short_Term_Flux'] = df_area['Carbonate_Sulfuric_Short_Term_Mass'] * 10**12 / site_area
    df_area['Carbonate_Sulfuric_Long_Term_Flux'] = df_area['Carbonate_Sulfuric_Long_Term_Mass'] * 10**12 / site_area
    df_area['Silicate_Carbonic_Short_Term_Flux'] = df_area['Silicate_Carbonic_Short_Term_Mass'] * 10**12 / site_area
    df_area['Silicate_Carbonic_Long_Term_Flux'] = df_area['Silicate_Carbonic_Long_Term_Mass'] * 10**12 / site_area
    df_area['Silicate_Sulfuric_Short_Term_Flux'] = df_area['Silicate_Sulfuric_Short_Term_Mass'] * 10**12 / site_area
    df_area['Silicate_Sulfuric_Long_Term_Flux'] = df_area['Silicate_Sulfuric_Long_Term_Mass'] * 10**12 / site_area
    
    
    ############################################################################################################
    
    flux_columns = [
        'Carbonate_Carbonic_Short_Term_Flux',
        'Carbonate_Carbonic_Long_Term_Flux',
        'Carbonate_Sulfuric_Short_Term_Flux',
        'Carbonate_Sulfuric_Long_Term_Flux',
        'Silicate_Carbonic_Short_Term_Flux',
        'Silicate_Carbonic_Long_Term_Flux',
        'Silicate_Sulfuric_Short_Term_Flux',
        'Silicate_Sulfuric_Long_Term_Flux'
    ]
    
    
    #Make a Total Short Term Flux:
    df_area['Total_Short_Term_Flux'] = df_area['Carbonate_Carbonic_Short_Term_Flux'] + df_area['Carbonate_Sulfuric_Short_Term_Flux'] + df_area['Silicate_Carbonic_Short_Term_Flux'] + df_area['Silicate_Sulfuric_Short_Term_Flux']
    
    #Make a Total Long Term Flux:
    df_area['Total_Long_Term_Flux'] = df_area['Carbonate_Carbonic_Long_Term_Flux'] + df_area['Carbonate_Sulfuric_Long_Term_Flux'] + df_area['Silicate_Carbonic_Long_Term_Flux'] + df_area['Silicate_Sulfuric_Long_Term_Flux']
    
    #Make a Net Long term Flux Budget:
    #df_area['Net_Flux_Budget'] = df_area['Total_Long_Term_Flux'] + df_area['Total_Short_Term_Flux']
    
    df_area['Net_Flux_Budget'] = df_area['Total_Long_Term_Flux']
    
    
    
    ############################################################################################################
    
    # Print the max and min values of the net flux:
    print('The maximum value of the net flux is:', df_area['Net_Flux_Budget'].max())
    print('The minimum value of the net flux before values less than -10^6 :', df_area['Net_Flux_Budget'].min())
    
    
    ############################################################################################################
    
    # Plot a histogram distribution of the net flux:
    plt.hist(df_area['Net_Flux_Budget'], bins=50)
    plt.title('Distribution of the Net Flux Budget')
    plt.xlabel('Net Flux Budget')
    plt.ylabel('Frequency')
    #plt.show()
    plt.close()
    
    ############################################################################################################
    
    # Filter out T7a-0722 - super negative
    df_area = df_area[df_area['unique_code'] != 'T7a-0722']
    
    # Filter out any values less than -10^6:
    df_area = df_area[df_area['Net_Flux_Budget'] > -10**7]
    
    # Print the max and min values of the net flux:
    print('The maximum value of the net flux is:', df_area['Net_Flux_Budget'].max())
    print('The minimum value of the net flux after values less than -10^6 :', df_area['Net_Flux_Budget'].min())
    
    
    ############################################################################################################
    
    
    # Export df_fraction to a excel file
    df_area.to_excel('/Users/enrico/Desktop/ROKOS Internship/QGIS/flux_areas.xlsx', index=False)
    
    return df_area
    
    