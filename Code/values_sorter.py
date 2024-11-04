
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





df = pd.read_excel('Datasets/Nepal Master Sheet.xlsx')


df_copy = df.copy()
    
df_copy = df_copy[df_copy['Sample type'] == 'Spring water']

df_copy['Na/Cl'] = df_copy['Na_molar'] / df_copy['Cl_molar']

print("Max 5 Na/Cl values:")
print(df_copy['Na/Cl'].nlargest(5))

print("\nMin 5 Na/Cl values:")
print(df_copy['Na/Cl'].nsmallest(5))

print("\nMean Na/Cl value:")
print(df_copy['Na/Cl'].mean())