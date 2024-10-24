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
# - Plots the NICB distribution using an interpolator to give a smooth curve. Bounds at +/-10%




def plot_NICB(df):
    """Plot NICB distribution"""
    # Plot the NICB distribution as a histogram
    fig, ax = plt.subplots()

    # Plot KDE
    (df['NICB']*100).plot.kde(ax=ax, legend=False, title='NICB distribution (%)')

    plt.axvline(x = 10, color = 'b', label = '10% Bounds')
    plt.axvline(x = -10, color = 'b')

    # Save plot
    plt.savefig('chemweathering/figures/NICB_distribution_percent.png')
    plt.close(fig)