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



########################### FUNCTION DEFINITION ###########################

# - Meant to be an external function called into Bathymetry to get a DEM there.
# - Was made when the code was still originally one long line, before we moved everything away into individual files
# - Not used anymore



# Define the path to the DEM (Digital Elevation Model) raster file
dem_path = 'chemweathering/data/output_SRTMGL1.tif'
output_path = os.path.join(os.getcwd(), dem_path)  # Adjusted path output

# Define bounds
west, south, east, north = -75.32426908093115, -13.526283371232736, -76.57091918634246, -11.749558766395864
bounds = (west - 0.05, south - 0.05, east + 0.05, north + 0.05)

# Calculate the center of the bounds
center_lon = (west + east) / 2
center_lat = (south + north) / 2

# Open the DEM raster file
with rasterio.open(output_path) as dem_raster:
    src_crs = dem_raster.crs
    
    # Clip the DEM to the specified bounds
    out_image, out_transform = mask(dem_raster, [shapely.geometry.box(*bounds)], crop=True)
    clipped_dem_array = out_image[0]

    # Calculate the destination transform based on the bounds and the shape of the clipped array
    dst_transform = from_bounds(*bounds, clipped_dem_array.shape[1], clipped_dem_array.shape[0])
    
    # Define the destination CRS
    dst_crs = 'EPSG:32718'

    # Get the dimensions of the clipped DEM
    height, width = clipped_dem_array.shape

    # Create the destination transform and array
    dst_transform, width, height = calculate_default_transform(
        src_crs, dst_crs, width, height, *bounds
    )
    dst_array = np.zeros((height, width), dtype=np.float32) #define x and y and z

    # Reproject the source data to the destination CRS
    reproject(
        source=clipped_dem_array,
        destination=dst_array,
        src_transform=out_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear
    )

# Filter out elevation values below 0
dst_array[dst_array < 0] = np.nan # nice shorthand
print(f"Reprojected DEM shape: {dst_array.shape}")

# Generate the x and y coordinates in the projected CRS
x = np.linspace(bounds[0], bounds[2], dst_array.shape[1])
y = np.linspace(bounds[1], bounds[3], dst_array.shape[0])
x, y = np.meshgrid(x, y)
z = dst_array

# Interpolator for DEM elevations
interpolator = RegularGridInterpolator((y[:, 0], x[0, :]), z, bounds_error=False, fill_value=np.nan)


# Create a 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
# Plot the surface
surf = ax.plot_surface(x, y, z, cmap='terrain', edgecolor='none')

plt.show()
sys.exit()