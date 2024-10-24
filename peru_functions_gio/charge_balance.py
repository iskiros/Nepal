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



######### CALCULATING NICB ##########


def charge_balance(df):
    """Function to charge balance"""
    cations = ['Ca', 'Mg', 'K', 'Na', 'Al', 'As', 'Ba', 'Bi', 'Fe', 'Li', 'Rb', 'Sr']
    anions = ['HCO3', 'Cl', 'SO4', 'H2PO4', 'CO3']

    # Calculate the sum of cations and anions
    df['sum_cations'] = df.loc[:, [col for col in df.columns if any(cation in col for cation in cations) and ' [aq] (meq/L)' in col]].sum(axis=1)

    df['sum_anions'] = df.loc[:, [col for col in df.columns if any(anion in col for anion in anions) and ' [aq] (meq/L)' in col]].sum(axis=1)

    # NICB calculation
    df['NICB diff'] = df['sum_cations'] - df['sum_anions']
    df['NICB sum'] = df['sum_cations'] + df['sum_anions']

    # Remove rows with NICB sum = 0
    df = df[df['NICB sum'] != 0].copy()

    # Use .loc to avoid the SettingWithCopyWarning
    df.loc[:, 'NICB'] = df['NICB diff'] / df['NICB sum']

    return df