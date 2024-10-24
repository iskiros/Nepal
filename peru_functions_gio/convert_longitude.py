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




########################### FUNCTION DEFINITION ############################


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
