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


##################################################################

######### OLD CODE USED FOR MAP MAKING IN 3D. NOT USED IN FINAL VERSION #########


def Bathymetry_Old(df):

    ####### FOLLOWED TUTORIAL ON HOW TO IMPORT A DEM INTO PYTHON USING RASTERIO #######

    # Copy the DataFrame to avoid modifying the original one
    df_copy = df.copy()
        
    # Define the path to the DEM (Digital Elevation Model) raster file
    dem_path = 'chemweathering/data/Raster.tif'
    output = os.getcwd() + '/' + dem_path  # Adjusted path output
    
    # Define bounds for clipping (actual coordinates)
    west, south, east, north = -76.0, -14.0, -77.0, -13.0  # Example bounds (replace with actual coordinates)
    
    west, south, east, north = bounds = west - .05, south - .05, east + .05, north + .05
    
    elevation.clip(bounds=bounds, output=output, product='SRTM3')

        
        # Execute elevation clipping (replace with actual function or process)
        # Replace this line with your actual method or library to clip
        # the elevation data to the specified bounds and save it to the output path.
        # If you are using a specific library or tool for this, replace accordingly.
        # For now, we'll set up the output path without the actual clipping.
        # elevation.clip(bounds=bounds, output=output, product='SRTM3')
        
    # Open the clipped DEM raster file (for illustration purposes)
    dem_raster = rasterio.open('.' + dem_path)
        
    src_crs = dem_raster.crs
    src_shape = src_height, src_width = dem_raster.shape
    src_transform = from_bounds(west, south, east, north, src_width, src_height)
    source = dem_raster.read(1)    
        

    # Define the destination CRS and transformation
    dst_crs = {'init': 'EPSG:32718'}
    dst_transform = from_origin(268000.0, 5207000.0, 250, 250)  # Adjusted coordinates
    dem_array = np.zeros((451, 623))
    dem_array[:] = np.nan
        
    # Initialize an empty array to store reprojected DEM data
    dem_array = np.zeros((src_height, src_width))
    dem_array[:] = np.nan
        
        # Reproject the source data to the destination CRS
    reproject(
        source=dem_raster.read(1),
        destination=dem_array,
        src_transform=dem_raster.transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear
        )
        
    # Load colormap for visualization
    topocmap = 'Spectral_r'
        
        # Define vmin and vmax for color mapping
    vmin = 180
    vmax = 575
    
    ###################################################################
    
    
    ####### PLOTTING THE TOPOGRAPHY DATA #######
    
    fig = plt.figure() 
    ax = fig.add_subplot(1, 1, 1) 
        
        # Plot the distribution of elevation data
    ax = sns.histplot(dem_array.ravel(), axlabel='Elevation (m)')
    ax = plt.gca()
        
        # Apply colormap and adjust alpha for patches
    #_ = [patch.set_color(topocmap(plt.Normalize(vmin=vmin, vmax=vmax)(patch.xy[0]))) for patch in ax.patches]
    #_ = [patch.set_alpha(1) for patch in ax.patches]
        
    # Save the figure
    ax.get_figure().savefig('chemweathering/data/hello.png')

