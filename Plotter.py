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





def Altitude_Alkalinity(df):
    # want to include only NICB valid samples. Takes only the unique code rows from dfneat if they also appear on unique_codes_valid
    
    df_copy = df.copy()
    
    # filter so df_copy only contains "Spring" in the "Type" column
    #df_copy = df_copy[df_copy['Sample Type'] == 'Spring']
    df_copy = df_copy[df_copy['Sample Type'] == 'Rain']

    


    ##################################################################


    plt.figure(figsize=(10,6))

    plt.scatter(df_copy['Elevation (m)'], df_copy['Alkalinity'],  alpha=0.7, s=70, c=df_copy['Longitude'], cmap='viridis')
    plt.xlabel('Elevation (m)')
    plt.ylabel('Alkalinity')
    #plt.axline((0, 0), (1, 1), linewidth=1, color='b', label='1 HCO3 to 1 SO4')
    #plt.axline((0, 0), (2, 1), linewidth=1, color='r', label='2 HCO3 to 1 SO4')
    plt.colorbar(label='Longitude') 
    plt.legend()
    #convert axes to Logarithmic scale
    #plt.xscale("log")
    #plt.yscale("log")
    for index, row in df_copy.iterrows():
        plt.text(row['Elevation (m)'], row['Alkalinity'], row['Sample#'], fontsize=8, ha='center', va='bottom')
 
    #plt.xlim(0.1,100)
    #plt.ylim(0.1,100)

    plt.title('Alkalinity vs Elevation')
    plt.show()
    #plt.savefig('nepal/elal.png')
    plt.close()
    
    
def Altitude_Temperature(df):
    # want to include only NICB valid samples. Takes only the unique code rows from dfneat if they also appear on unique_codes_valid
    
    df_copy = df.copy()
    
    # filter so df_copy only contains "Spring" in the "Type" column
    #df_copy = df_copy[df_copy['Sample Type'] == 'Spring']
    df_copy = df_copy[df_copy['Sample Type'] == 'Rain']

    


    ##################################################################


    plt.figure(figsize=(10,6))

    plt.scatter(df_copy['Elevation (m)'], df_copy['T'],  alpha=0.7, s=70, c=df_copy['Longitude'], cmap='viridis')
    plt.xlabel('Elevation (m)')
    plt.ylabel('Temperature')
    #plt.axline((0, 0), (1, 1), linewidth=1, color='b', label='1 HCO3 to 1 SO4')
    #plt.axline((0, 0), (2, 1), linewidth=1, color='r', label='2 HCO3 to 1 SO4')
    plt.colorbar(label='Longitude') 
    plt.legend()
    #convert axes to Logarithmic scale
    #plt.xscale("log")
    #plt.yscale("log")
    for index, row in df_copy.iterrows():
        plt.text(row['Elevation (m)'], row['T'], row['Sample#'], fontsize=8, ha='center', va='bottom')
 
    #plt.xlim(0.1,100)
    #plt.ylim(0.1,100)

    plt.title('Temperature vs Elevation')
    plt.show()
    #plt.savefig('nepal/eltemp.png')
    plt.close()    
    
    
    
def Altitude_pH(df):
    # want to include only NICB valid samples. Takes only the unique code rows from dfneat if they also appear on unique_codes_valid
    
    df_copy = df.copy()
    
    # filter so df_copy only contains "Spring" in the "Type" column
    #df_copy = df_copy[df_copy['Sample Type'] == 'Spring']
    df_copy = df_copy[df_copy['Sample Type'] == 'Rain']

    


    ##################################################################


    plt.figure(figsize=(10,6))

    plt.scatter(df_copy['Elevation (m)'], df_copy['pH'],  alpha=0.7, s=70, c=df_copy['Longitude'], cmap='viridis')
    plt.xlabel('Elevation (m)')
    plt.ylabel('pH')
    #plt.axline((0, 0), (1, 1), linewidth=1, color='b', label='1 HCO3 to 1 SO4')
    #plt.axline((0, 0), (2, 1), linewidth=1, color='r', label='2 HCO3 to 1 SO4')
    plt.colorbar(label='Longitude') 
    plt.legend()
    #convert axes to Logarithmic scale
    #plt.xscale("log")
    #plt.yscale("log")
    for index, row in df_copy.iterrows():
        plt.text(row['Elevation (m)'], row['pH'], row['Sample#'], fontsize=8, ha='center', va='bottom')
 
    #plt.xlim(0.1,100)
    #plt.ylim(0.1,100)

    plt.title('pH vs Elevation')
    plt.show()
    #plt.savefig('nepal/eltemp.png')
    plt.close()     
    
    
def Altitude_TDS(df):
    # want to include only NICB valid samples. Takes only the unique code rows from dfneat if they also appear on unique_codes_valid
    
    df_copy = df.copy()
    
    # filter so df_copy only contains "Spring" in the "Type" column
    #df_copy = df_copy[df_copy['Sample Type'] == 'Spring']
    df_copy = df_copy[df_copy['Sample Type'] == 'Rain']

    


    ##################################################################


    plt.figure(figsize=(10,6))

    plt.scatter(df_copy['Elevation (m)'], df_copy['TDS'],  alpha=0.7, s=70, c=df_copy['Longitude'], cmap='viridis')
    plt.xlabel('Elevation (m)')
    plt.ylabel('TDS')
    #plt.axline((0, 0), (1, 1), linewidth=1, color='b', label='1 HCO3 to 1 SO4')
    #plt.axline((0, 0), (2, 1), linewidth=1, color='r', label='2 HCO3 to 1 SO4')
    plt.colorbar(label='Longitude') 
    plt.legend()
    #convert axes to Logarithmic scale
    #plt.xscale("log")
    #plt.yscale("log")
    for index, row in df_copy.iterrows():
        plt.text(row['Elevation (m)'], row['TDS'], row['Sample#'], fontsize=8, ha='center', va='bottom')
 
    #plt.xlim(0.1,100)
    #plt.ylim(0.1,100)

    plt.title('TDS vs Elevation')
    plt.show()
    #plt.savefig('nepal/eltemp.png')
    plt.close()     




def Alkalinity_TDS(df):
    # want to include only NICB valid samples. Takes only the unique code rows from dfneat if they also appear on unique_codes_valid
    
    df_copy = df.copy()
    
    # filter so df_copy only contains "Spring" in the "Type" column
    #df_copy = df_copy[df_copy['Sample Type'] == 'Spring']
    df_copy = df_copy[df_copy['Sample Type'] == 'Rain']

    


    ##################################################################


    plt.figure(figsize=(10,6))

    plt.scatter(df_copy['Alkalinity'], df_copy['TDS'],  alpha=0.7, s=70, c=df_copy['Longitude'], cmap='viridis')
    plt.xlabel('Alkalinity')
    plt.ylabel('TDS')
    #plt.axline((0, 0), (1, 1), linewidth=1, color='b', label='1 HCO3 to 1 SO4')
    #plt.axline((0, 0), (2, 1), linewidth=1, color='r', label='2 HCO3 to 1 SO4')
    plt.colorbar(label='Longitude') 
    plt.legend()
    #convert axes to Logarithmic scale
    #plt.xscale("log")
    #plt.yscale("log")
    for index, row in df_copy.iterrows():
        plt.text(row['Alkalinity'], row['TDS'], row['Sample#'], fontsize=8, ha='center', va='bottom')
 
    #plt.xlim(0.1,100)
    #plt.ylim(0.1,100)

    plt.title('TDS vs Alkalinity')
    plt.show()
    #plt.savefig('nepal/eltemp.png')
    plt.close()     

    
df = pd.read_excel('points1024.xlsx')
    
    
Altitude_Alkalinity(df)
Altitude_Temperature(df)
Altitude_pH(df)
Altitude_TDS(df)


Alkalinity_TDS(df)