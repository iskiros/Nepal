import sys
import os
import math
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
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
from geopy.distance import geodesic
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


# Define endmember points in 3D space
points3 = {
    "LH": np.array([1/0.38, 0.783, 1]),
    "HHC": np.array([1/0.23, 0.739, 2]),
    "TSS": np.array([1/6.13, 0.717, 3]),
    "RAIN": np.array([130, 0.709041454178464, 0.7])
}

# Convert to numpy array for easier processing
endmember_names = list(points3.keys())
endmember_positions = np.array([points3[name] for name in endmember_names])

# 3D Plot of Tetrahedron and Sample Points
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot endmembers
ax.scatter(
    endmember_positions[:, 0], endmember_positions[:, 1], endmember_positions[:, 2],
    color='red', s=100, label="Endmembers"
)

# Annotate endmembers
for name, pos in points3.items():
    ax.text(pos[0], pos[1], pos[2], name, fontsize=10, fontweight='bold')

# Define tetrahedron edges
edges = [
    ("LH", "HHC"), ("LH", "TSS"), ("LH", "RAIN"),
    ("HHC", "TSS"), ("HHC", "RAIN"), ("TSS", "RAIN")
]

# Plot edges of the tetrahedron
for edge in edges:
    point1, point2 = points3[edge[0]], points3[edge[1]]
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], 'k--')


samples = pd.read_excel("/Users/enrico/Desktop/Part III Project/Code/Nepal-1/Code/samples_endmember_sr_analysis.xlsx")

samples = samples.dropna()  # Remove NaN values
samples = samples[samples['Sr_mM'] > 0]  # Remove zero values to avoid division errors
samples['1/Sr_uM'] = 1 / (samples['Sr_mM'] * 1000)
samples = samples.replace([np.inf, -np.inf], np.nan).dropna()  # Remove inf values


# Extract sample points for plotting
sample_points = samples[['1/Sr_uM', 'Sr87/Sr86', '1000Sr/Ca']].values

# Plot sample points
ax.scatter(
    sample_points[:, 0], sample_points[:, 1], sample_points[:, 2],
    color='blue', s=50, label="Samples"
)

# Annotate sample points
for i, sample in enumerate(sample_points):
    ax.text(sample[0], sample[1], sample[2], f"Sample {i+1}", fontsize=9)

# Set axis labels
ax.set_xlabel("1/Sr (uM⁻¹)")
ax.set_ylabel("87Sr/86Sr")
ax.set_zlabel("Sr/Ca")
ax.set_title("3D Tetrahedron of Endmembers and Sample Points")

plt.legend()
plt.show()
