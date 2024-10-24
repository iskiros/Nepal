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
# - Calculates which values are within 10% +/-, and returns a new dataframe with unique codes which are valid and invalid
# - Used everywhere




def NICB_Valid(df_neat):
    unique_codes_valid = []
    unique_codes_invalid = []
    nicb_valid = []
    nicb_invalid = []

    #Iterate through NICB values
    for index, i in df_neat['NICB'].items():
        unique_code = df_neat.loc[index, 'unique_code']
        if i > 0.1 or i < -0.1:
            nicb_invalid.append(i)
            unique_codes_invalid.append(unique_code)
        else:
            nicb_valid.append(i)
            unique_codes_valid.append(unique_code)

    # Calculate the maximum length to pad with None
    max_length = max(len(nicb_valid), len(nicb_invalid))

    # Pad lists with None to match the maximum length
    nicb_valid += [None] * (max_length - len(nicb_valid))
    nicb_invalid += [None] * (max_length - len(nicb_invalid))
    unique_codes_valid += [None] * (max_length - len(unique_codes_valid))
    unique_codes_invalid += [None] * (max_length - len(unique_codes_invalid))

    # Create the new DataFrame
    NICB_Balance = pd.DataFrame({
        'unique_code_valid': unique_codes_valid,
        'NICB_Valid': nicb_valid,
        'unique_code_invalid': unique_codes_invalid,
        'NICB_Invalid': nicb_invalid
    })  
    
    #print(len(NICB_Balance['unique_code_valid']))
            
    NICB_Balance.to_csv('chemweathering/data/ValidNICB.csv', index=False)

    return (NICB_Balance)



