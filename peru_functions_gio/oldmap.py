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
# - Previously used to plot html data. First attempt
# - Not used since then



# def oldmap(df):
    
#     m = folium.Map(location=[df.latitude_converted.mean(), df.longitude_converted.mean()], zoom_start=10)

#     tile = folium.TileLayer(
#         tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
#         attr='Esri',
#         name='Esri Satellite',
#         overlay=False,
#         control=True
#     ).add_to(m)

# # Define the colormap correctly
#     # Define the logarithmic colormap (using Viridis)
#     log_cmap = LogColormap(colors=['blue', 'green', 'yellow'], vmin=0.1, vmax=100, index=[0.1, 1, 10, 100], caption='Logarithmic Colormap')

#     # Add the colormap to the map
#     log_cmap.add_to(m)

#     legend_html = '''
#          <div style="position: fixed; 
#                      top: 10px; right: 25px; width: 500px; height: 40px; 
#                      border:2px solid grey; z-index:9999; font-size:14px;
#                      background-color: white; opacity: 0.5;
#                     "> 
#         &nbsp; <strong>Ca/Na</strong> <br>
#         &nbsp; 0 - 100 <br>
#             </div>
#         '''

#     m.get_root().html.add_child(folium.Element(legend_html))
    
#     #df['CaNa'] = df['Ca [aq] (mM)']/df['Na [aq] (mM)']

#     for index, location_info in df.iterrows():
#         value = location_info['CaNa']
#     # Ensure the value is within the range of the colormap
#         if 0 <= value <= 100:
#             folium.CircleMarker(
#                 [location_info["latitude_converted"], location_info["longitude_converted"]],
#                 popup=f"Ca/Na: {location_info['CaNa']}<br> <br> Sample: {location_info['unique_code']}",
#                 radius=10,
#                 color="black",
#                 weight=1,
#                 fill_opacity=0.6,
#                 opacity=1,
#                 fill_color=log_cmap(value),
#                 fill=True
#             ).add_to(m)


# #print((data_locations['SampleID']))

# # Save the map to the specified folder
#     folder_path = 'chemweathering/data/figures'
#     file_name = 'mapgio.html'
#     os.makedirs(folder_path, exist_ok=True)
#     full_path = os.path.join(folder_path, file_name)
#     m.save(full_path)
