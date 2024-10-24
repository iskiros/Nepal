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
# - Selects relevant data from the raw dataframe. Columns selected are major elements, pH, altitude, and water body etc



def slice_relevant_data(df):
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
    additional_columns = ['longitude_new', 'longitude_old', 'latitude_new', 'latitude_old', 'calculated_discharge_in_m3_s-1', 'Alkalinity_CaCO3_in_mg_kg-1', 'field_pH', 'altitude_in_m', 'water_body']  # Add more column names as needed
    major_elements += additional_columns

    # slice the dataframe to include only the major elements
    df_slice = df[major_elements]
    


    # append the sample ID to the sliced dataframe
    df_slice.insert(0, 'unique_code', df['unique_code'])
    
    #print(df_slice.columns)
    
    #The first argument, 0, specifies the index position where the new column should be inserted. 
    # In this case, it is inserted at the beginning of the DataFrame.

    #The second argument, 'unique_code', is the name of the new column that will be inserted.

    #The third argument, df['unique_code'], is the data that will be populated in the new column. 
    # It is taken from the 'unique_code' column of the df DataFrame.
    
    

    #print(df_slice.columns)
    
    #print(df_slice)

    return df_slice