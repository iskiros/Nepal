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
import PyCO2SYS as cs
import matplotlib.colors as mcolors

    
    
    
df = pd.read_excel('Datasets/Nepal Master Sheet2.xlsx')
# Define a function to calculate DIC for each row
def calculate_dic(row):
    # Create a dictionary with the input parameters
    params = {
        'par1': row['Alkalinity'],     # Total Alkalinity
        'par2': row['pH'],             # pH
        'par1_type': 1,                # Alkalinity type
        'par2_type': 3,                # pH on the total scale
        'temperature': row['Temperature'], # Input temperature
        'pressure': 0,                 # Set pressure to 0 for surface (change if depth is known)
    }
    
    # Calculate the CO2 system
    results = cs.sys(**params)
    
    # Extract DIC
    return results['dic']


# Apply the function to each row
df['DIC'] = df.apply(calculate_dic, axis=1)

df.to_excel('Datasets/Nepal Master Sheet_DIC.xlsx', index=False)
    