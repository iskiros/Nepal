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

    #print(df_copy.columns)

    return df_copy


def chloride_correction(df):
    rain_values = {
        'Ca/Cl': 2.97,
        'Mg/Cl': 1.99,
        'K/Cl': 0.72,
        'Na/Cl': 1.35,
        'Si/Cl': 0.41,
        'Sr/Cl': 4.05e-3
    }

    df_copy = df.copy()

    # Iterate over each element in the rain_values dictionary
    for element_ratio, ratio_value in rain_values.items():
        element = element_ratio.split('/')[0] + ' (uM)'
        
        # Lowest Cl value in the river:
        riverlowestCl = df_copy['Cl_molar'].min()
        
        if element in df_copy.columns:
            # Perform the correction calculation
            df_copy['*' + element] = df_copy[element] - ratio_value * riverlowestCl
        else:
            # Print a message if the current element is not found in the df_copy DataFrame
            print(f"Element {element} not found in df_copy")

    return df_copy



def XSil(df):
    
    df_copy = df.copy()
    
    #Gaillardet's ratios
    #We assign the following chemical composition to the silicate end member: 
    # WE can refine this with our own data
    
    #Ca/Na = 0.35
    #Mg/Na = 0.24
    #HCO3/Na =  2
    #1000*Sr/Na = 3
    
    CaNa_sil = 0.35
    
    MgNa_sil = 0.24

    df_copy['Ca_Sil'] = df_copy['*Na (uM)'] * CaNa_sil
    
    df_copy['Mg_Sil'] = df_copy['*Na (uM)'] * MgNa_sil
    
    df_copy['X_Sil'] = ((2*df_copy['Ca_Sil']) + (2*df_copy['Mg_Sil']) + df_copy['*K (uM)'] + df_copy['*Na (uM)'])/((2*df_copy['*Ca (uM)']) + (2*df_copy['*Ca (uM)']) + df_copy['*K (uM)'] + df_copy['*Na (uM)'])
    
    
    
    return df_copy
    
    
    
    
    

def main():
    
    
    df = pd.read_excel('Datasets/Nepal Master Sheet.xlsx')
    
    
    # filter data for sample type = spring water
    df = df[df['Sample type'] == 'Spring water']
    
    # calculate molar concentrations
    df_molar = calculate_molar_concentration(df)
    
    df_corrected = chloride_correction(df_molar)
    
    #print(df_corrected.head())
    
    df_XSil = XSil(df_corrected)
    
    print(df_XSil.head())
    
    
    
    
    # filtered_df = df_XSil[df_XSil['X_Sil'] > 1.0]
    # print("These samples had XSil >1:")
    # for index, row in filtered_df.iterrows():
    #     print(row['Sample ID'])
    
    
    # Plot XSil against elevation:
    
    # Plot XSil against elevation
    fig, ax = plt.subplots()
    ax.scatter(df_XSil['X_Sil'], df_XSil['Sr87/Sr86'])
    ax.set_xlabel('XSil')
    ax.set_ylabel('Sr87/Sr86')
    ax.set_title('Sr87/Sr86 vs XSil')
    #plt.show()
    
    df_XSil['Si/Ca'] = df_XSil['Si (uM)'] / df_XSil['Ca (uM)']
    df_XSil['Na/Ca'] = df_XSil['Na (uM)'] / df_XSil['Ca (uM)']
    
    fig, ax = plt.subplots()
    ax.scatter(df_XSil['Na/Ca'], df_XSil['Si/Ca'])
    ax.set_xlabel('Na/Ca')
    ax.set_ylabel('Si/Ca')
    ax.set_title('Na/Ca vs Si/Ca')
    plt.show()
    
    






if __name__ == '__main__':
    main()