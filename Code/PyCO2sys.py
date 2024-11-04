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
from matplotlib.backends.backend_pdf import PdfPages
import streamlit as st
import matplotlib.colors as mcolors
import PyCO2SYS as cs


  
df = pd.read_excel('Datasets/Nepal Master Sheet.xlsx', sheet_name='Final_compiled')
    




def calculate_bicarbonate_carbonate(row):
    # Create a dictionary with the input parameters
    params = {
        'par1': row['Alkalinity'],     # Total Alkalinity
        'par2': row['pH'],             # pH
        'par1_type': 1,                # Alkalinity type
        'par2_type': 3,                # pH on the total scale
        'temperature': row['Temperature'], # Input temperature in Celsius
        #'salinity': row.get('Salinity', 35), # Default to 35 if salinity is missing
        'pressure': 0,                 # Set pressure to 0 for surface (change if depth is known)
    }
    
    # Calculate the CO2 system
    results = cs.sys(**params)
    


    # Check the available keys in results to ensure we use the correct ones
    #print("Available keys in results:", results.keys())

    
    # Extract bicarbonate and carbonate ion concentrations
    bicarbonate = results['HCO3'] / 1000  # in µmol/kg
    carbonate = results['CO3'] / 1000      # in µmol/kg
    
    return pd.Series({'HCO3[mM]': bicarbonate, 'CO3[mM]': carbonate})

# Apply the function to each row
df[['HCO3[mM]', 'CO3[mM]']] = df.apply(calculate_bicarbonate_carbonate, axis=1)

df.to_excel('Datasets/Nepal Master Sheet.xlsx', index=False)
