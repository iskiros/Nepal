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
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcolors





def Altitude_Alkalinity(df):
    # want to include only NICB valid samples. Takes only the unique code rows from dfneat if they also appear on unique_codes_valid
    df_copy = df.copy()
    
    # filter so df_copy only contains "Spring" in the "Type" column
    #df_copy = df_copy[df_copy['Sample Type'] == 'Spring']
    df_copy = df_copy[df_copy['Sample type'] == 'Spring']
    
    # print length of df_copy
    print(len(df_copy))

    fig, axs = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Various Parameters vs Elevation and Alkalinity')

    # Define a color map for the seasons
    season_colors = {
        'Nov_22': 'blue',
        'Apr_23': 'green',
        'Oct_23': 'red',
        'Sep_24': 'purple'
    }

    # Create a PdfPages object to save multiple plots in a single PDF
    pdf_pages = PdfPages('RainPlots.pdf')

    # Alkalinity vs Elevation
    for season, color in season_colors.items():
        season_data = df_copy[df_copy['Season'] == season]
        axs[0, 0].scatter(season_data['Elevation'], season_data['Alkalinity'], alpha=0.7, s=70, color=color, label=season)
    axs[0, 0].set_xlabel('Elevation')
    axs[0, 0].set_ylabel('Alkalinity')
    axs[0, 0].set_title('Alkalinity vs Elevation')
    for index, row in df_copy.iterrows():
        axs[0, 0].text(row['Elevation'], row['Alkalinity'], row['Sample ID'], fontsize=8, ha='center', va='bottom')

    # Temperature vs Elevation
    for season, color in season_colors.items():
        season_data = df_copy[df_copy['Season'] == season]
        axs[0, 1].scatter(season_data['Elevation'], season_data['Temperature'], alpha=0.7, s=70, color=color, label=season)
    axs[0, 1].set_xlabel('Elevation')
    axs[0, 1].set_ylabel('Temperature')
    axs[0, 1].set_title('Temperature vs Elevation')
    for index, row in df_copy.iterrows():
        axs[0, 1].text(row['Elevation'], row['Temperature'], row['Sample ID'], fontsize=8, ha='center', va='bottom')

    # pH vs Elevation
    for season, color in season_colors.items():
        season_data = df_copy[df_copy['Season'] == season]
        axs[1, 0].scatter(season_data['Elevation'], season_data['pH'], alpha=0.7, s=70, color=color, label=season)
    axs[1, 0].set_xlabel('Elevation')
    axs[1, 0].set_ylabel('pH')
    axs[1, 0].set_title('pH vs Elevation')
    for index, row in df_copy.iterrows():
        axs[1, 0].text(row['Elevation'], row['pH'], row['Sample ID'], fontsize=8, ha='center', va='bottom')

    # TDS vs Elevation
    for season, color in season_colors.items():
        season_data = df_copy[df_copy['Season'] == season]
        axs[1, 1].scatter(season_data['Elevation'], season_data['TDS'], alpha=0.7, s=70, color=color, label=season)
    axs[1, 1].set_xlabel('Elevation')
    axs[1, 1].set_ylabel('TDS')
    axs[1, 1].set_title('TDS vs Elevation')
    for index, row in df_copy.iterrows():
        axs[1, 1].text(row['Elevation'], row['TDS'], row['Sample ID'], fontsize=8, ha='center', va='bottom')

    # TDS vs Alkalinity
    for season, color in season_colors.items():
        season_data = df_copy[df_copy['Season'] == season]
        axs[2, 0].scatter(season_data['Alkalinity'], season_data['TDS'], alpha=0.7, s=70, color=color, label=season)
    axs[2, 0].set_xlabel('Alkalinity')
    axs[2, 0].set_ylabel('TDS')
    axs[2, 0].set_title('TDS vs Alkalinity')
    for index, row in df_copy.iterrows():
        axs[2, 0].text(row['Alkalinity'], row['TDS'], row['Sample ID'], fontsize=8, ha='center', va='bottom')


   
    # Hide the empty subplot (bottom right)
    fig.delaxes(axs[2, 1])

    # Add legend for seasons
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig('RainPlots.png')
    plt.show()

    plt.close()


    import matplotlib.pyplot as plt
def Altitude_Alkalinity_PDF(df):
    df_copy = df.copy()
    
    # Filter so df_copy only contains "Spring" in the "Sample type" column
    df_copy = df_copy[df_copy['Sample type'] == 'Spring']
    
    # Print length of df_copy
    print(len(df_copy))

    fig, axs = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Various Parameters vs Elevation and Alkalinity')

    # Define a color map for the seasons
    season_colors = {
        'Nov_22': 'blue',
        'Apr_23': 'green',
        'Oct_23': 'red',
        'Sep_24': 'purple'
    }

    # Create a PdfPages object to save multiple plots in a single PDF
    with PdfPages('RainPlots.pdf') as pdf_pages:
        
        # Alkalinity vs Elevation
        fig, ax = plt.subplots(figsize=(8, 6))
        for season, color in season_colors.items():
            season_data = df_copy[df_copy['Season'] == season]
            ax.scatter(season_data['Elevation'], season_data['Alkalinity'], alpha=0.7, s=70, color=color, label=season)
        ax.set_xlabel('Elevation')
        ax.set_ylabel('Alkalinity')
        ax.set_title('Alkalinity vs Elevation')
        for index, row in df_copy.iterrows():
            ax.text(row['Elevation'], row['Alkalinity'], row['Sample ID'], fontsize=8, ha='center', va='bottom')
        ax.legend()
        pdf_pages.savefig(fig)  # Save the current plot as a separate page in the PDF
        plt.close(fig)

        # Temperature vs Elevation
        fig, ax = plt.subplots(figsize=(8, 6))
        for season, color in season_colors.items():
            season_data = df_copy[df_copy['Season'] == season]
            ax.scatter(season_data['Elevation'], season_data['Temperature'], alpha=0.7, s=70, color=color, label=season)
        ax.set_xlabel('Elevation')
        ax.set_ylabel('Temperature')
        ax.set_title('Temperature vs Elevation')
        for index, row in df_copy.iterrows():
            ax.text(row['Elevation'], row['Temperature'], row['Sample ID'], fontsize=8, ha='center', va='bottom')
        ax.legend()
        pdf_pages.savefig(fig)  # Save the current plot as a separate page in the PDF
        plt.close(fig)

        # pH vs Elevation
        fig, ax = plt.subplots(figsize=(8, 6))
        for season, color in season_colors.items():
            season_data = df_copy[df_copy['Season'] == season]
            ax.scatter(season_data['Elevation'], season_data['pH'], alpha=0.7, s=70, color=color, label=season)
        ax.set_xlabel('Elevation')
        ax.set_ylabel('pH')
        ax.set_title('pH vs Elevation')
        for index, row in df_copy.iterrows():
            ax.text(row['Elevation'], row['pH'], row['Sample ID'], fontsize=8, ha='center', va='bottom')
        ax.legend()
        pdf_pages.savefig(fig)  # Save the current plot as a separate page in the PDF
        plt.close(fig)

        # TDS vs Elevation
        fig, ax = plt.subplots(figsize=(8, 6))
        for season, color in season_colors.items():
            season_data = df_copy[df_copy['Season'] == season]
            ax.scatter(season_data['Elevation'], season_data['TDS'], alpha=0.7, s=70, color=color, label=season)
        ax.set_xlabel('Elevation')
        ax.set_ylabel('TDS')
        ax.set_title('TDS vs Elevation')
        for index, row in df_copy.iterrows():
            ax.text(row['Elevation'], row['TDS'], row['Sample ID'], fontsize=8, ha='center', va='bottom')
        ax.legend()
        pdf_pages.savefig(fig)  # Save the current plot as a separate page in the PDF
        plt.close(fig)

        # TDS vs Alkalinity
        fig, ax = plt.subplots(figsize=(8, 6))
        for season, color in season_colors.items():
            season_data = df_copy[df_copy['Season'] == season]
            ax.scatter(season_data['Alkalinity'], season_data['TDS'], alpha=0.7, s=70, color=color, label=season)
        ax.set_xlabel('Alkalinity')
        ax.set_ylabel('TDS')
        ax.set_title('TDS vs Alkalinity')
        for index, row in df_copy.iterrows():
            ax.text(row['Alkalinity'], row['TDS'], row['Sample ID'], fontsize=8, ha='center', va='bottom')
        ax.legend()
        pdf_pages.savefig(fig)  # Save the current plot as a separate page in the PDF
        plt.close(fig)
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





def Ratios(df):
    
    df_copy = df.copy()
    
    # Filter so df_copy only contains "Spring" in the "Sample type" column
    df_copy = df_copy[df_copy['Sample type'] == 'Spring water']
    
    # Print length of df_copy
    print(len(df_copy))

    # Define a color map for the seasons
    season_colors = {
        'Nov_22': 'blue',
        'Apr_23': 'green',
        'Oct_23': 'red',
        'Sep_24': 'purple'
    }
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for season, color in season_colors.items():
        season_data = df_copy[df_copy['Season'] == season]
        ax.scatter(season_data['Elevation'], season_data['Sr/Ca'], alpha=0.7, s=70, color=color, label=season)
    ax.set_xlabel('Elevation')
    ax.set_ylabel('Sr/Ca')
    ax.set_title('Sr/Ca vs Elevation')
    
    ax.legend()
    plt.show()
    plt.close(fig)





    
df = pd.read_excel('Datasets/Nepal Master Sheet.xlsx', sheet_name='Final_compiled')
    

#Altitude_Alkalinity_PDF(df)
#Altitude_Temperature(df)
#Altitude_pH(df)
#Altitude_TDS(df)
#Alkalinity_TDS(df)


Ratios(df)  