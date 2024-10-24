import sys
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import geodatasets
import geopandas as gpd
import re
import folium
import earthpy as et
import webbrowser
import statsmodels.api as sm
import elevation
import shapely.geometry
import seaborn as sns
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
import rasterio
import earthpy.spatial as es
import matplotlib.colors as mcolors
import matplotlib
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
from matplotlib.colors import ListedColormap, Normalize
from scipy.interpolate import griddata



def calculate_molar_concentration(df):
    """Calculate molar concentration of the major elements"""
    element_dict = {
        'molecules': ['Al', 'Ba', 'Ca', 'Fe', 'K', 'Li', 'Mg', 'Mn', 'Na', 'S', 'Si', 'Sr'],
        'molar_mass': [26.98, 137.33, 40.08, 55.85, 39.10, 6.94, 24.31, 54.94, 22.99, 32.06, 28.09, 87.62],
        'valency': [3, 2, 2, 2, 1, 1, 2, 2, 1, 2, 4, 2]
    }

    # Suffixes
    suffix = '_ppm'

    df_copy = df.copy()

    for element, molar_mass, valency in zip(element_dict['molecules'],
                                            element_dict['molar_mass'],
                                            element_dict['valency']):
        # Check if the standard or alternate column exists
        if element + suffix in df_copy.columns:
            col_name = element + suffix
        else:
            # Skip element if neither column exists
            continue

        # Create new molar columns
        df_copy.loc[:, element + ' (uM)'] = df_copy[col_name] / molar_mass * 1000
        df_copy.loc[:, element + ' (ueq/L)'] = df_copy.loc[:, element + ' (uM)'] * valency * 1000

    print(df_copy.columns)

    return df_copy


def chloride_correction(df):
    
    rain_df = pd.read_excel('/Users/enrico/Downloads/Nepal Master Sheet.xlsx', sheet_name='Rainwater') ### FOR EXAMPLE

    df_copy = df.copy()

    # Iterate over each element (column) in the rain_processed DataFrame
    for element in rain_df.columns:
        
        if ' (uM)' in element:
            
            if element in df_copy.columns:
                
                
                # Get the value of the current element from the rain_processed DataFrame (assuming there's only one value)
                
                for i, sample in rain_df.iterrows():
                    
                    rain_element_value = sample[element]
                    
                    # Get the value of 'Cl [aq] (mM)' from the rain_processed DataFrame (assuming there's only one value)
                    
                    cl_rain_value = sample['Cl (uM)']
                    
                    # Perform the correction calculation:
                    # For each row in df_copy, subtract the rain element value from the river element value
                    
                    df_copy['*' + str(sample['rain_sample_id']) + element] = df_copy[element] - rain_element_value / cl_rain_value * df_copy['Cl (uM)']
            else:
                # Print a message if the current element is not found in the df_copy DataFrame
                print(f"Element {element} not found in df_copy")




def main():
    
    
    df = pd.read_excel('/Users/enrico/Downloads/Nepal Master Sheet.xlsx', sheet_name='Final_compiled')
    
    # filter data for sample type = spring water
    df = df[df['Sample type'] == 'Spring water']
    
    # calculate molar concentrations
    df_molar = calculate_molar_concentration(df)
    
    
    






if __name__ == '__main__':
    main()