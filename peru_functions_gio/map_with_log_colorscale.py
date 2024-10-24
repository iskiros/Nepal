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


def convert_latitude(row):
    if pd.notna(row['latitude_new']):
        return row['latitude_new']
    elif pd.notna(row['latitude_old']):
        match = re.match(r'(\d+)\s+(\d+\.\d+)', str(row['latitude_old']))
        if match:
            degrees = int(match.group(1))
            minutes = float(match.group(2))
            decimal_degrees = -(degrees + (minutes / 60))
            return decimal_degrees
    return None


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




def map_with_log_colorscale(df):
    

    df['CaNa'] = df['Ca [aq] (mM)']/df['Na [aq] (mM)']

    ###### Above Potentially removable if you merge the datasets with GDPlot
    
    
    
    
    
    
    
    m = folium.Map(location=[df.latitude_converted.mean(), df.longitude_converted.mean()], zoom_start=10)

    # Add Esri Satellite tile layer
    tile = folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Satellite',
        overlay=False,
        control=True
    ).add_to(m)

    # Define logarithmic color scale using matplotlib
    cmap = get_cmap('viridis')  # Choose colormap, e.g., Viridis
    norm = LogNorm(vmin=0.1, vmax=100)  # Adjust vmin and vmax based on your data range and logarithmic scale

    # Iterate through the dataframe and add CircleMarkers to the map
    for index, location_info in df.iterrows():
        value = location_info['CaNa']
        
        # Ensure the value is within the range of the colormap
        if 0.1 <= value <= 100:
            color = cmap(norm(value))  # Get color from colormap based on logarithmic scale
            color_hex = plt.cm.colors.rgb2hex(color)
            
            folium.CircleMarker(
                [location_info["latitude_converted"], location_info["longitude_converted"]],
                popup=f"Ca/Na: {location_info['CaNa']}<br><br>Sample: {location_info['unique_code']}",
                radius=10,
                color="black",
                weight=1,
                fill_opacity=0.8,
                opacity=1,
                fill_color=color_hex,
                fill=True
            ).add_to(m)

    # Legend HTML
       # Legend HTML
    legend_html = '''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 280px; height: 100px; 
                border: 2px solid grey; z-index: 9999; font-size: 14px;
                background-color: white; opacity: 0.9; padding: 10px;">
    <strong>Logarithmic Colormap, Ca/Na</strong><br>
    <text x="10" y="50" text-anchor="start" fill="black">&emsp;&emsp;0.1&emsp;&emsp;&emsp; 1&emsp;&emsp;&emsp;&emsp; 10&emsp;&emsp;&emsp; 100</text>
    <svg height="30" width="250">
      <rect x="0" y="0" width="250" height="30" style="fill:rgb(255,255,255);stroke-width:1;stroke:rgb(0,0,0)" />
      <rect x="5" y="5" width="50" height="20" style="fill:{color_min}" />
      <rect x="65" y="5" width="50" height="20" style="fill:{color_mid1}" />
      <rect x="125" y="5" width="50" height="20" style="fill:{color_mid2}" />
      <rect x="185" y="5" width="50" height="20" style="fill:{color_max}" />
    </svg>
    </div>
    '''.format(color_min=plt.cm.colors.rgb2hex(cmap(norm(0.1))),
               color_mid1=plt.cm.colors.rgb2hex(cmap(norm(1))),
               color_mid2=plt.cm.colors.rgb2hex(cmap(norm(10))),
               color_max=plt.cm.colors.rgb2hex(cmap(norm(100))))
    
    m.get_root().html.add_child(folium.Element(legend_html))


    # Save the map to the specified folder
    folder_path = 'chemweathering/data/figures'
    file_name = 'map_logcana_gio.html'
    os.makedirs(folder_path, exist_ok=True)
    full_path = os.path.join(folder_path, file_name)
    m.save(full_path)

