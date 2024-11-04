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
import streamlit as st
import matplotlib.colors as mcolors
import PyCO2SYS as cs





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
    
     # Filter so df_copy only contains "Spring" in the "Sample type" column
    df_copy = df_copy[df_copy['Sample type'] == 'Spring water']
    
    # filter so df_copy only contains "Spring" in the "Type" column
    #df_copy = df_copy[df_copy['Sample Type'] == 'Spring']
    #df_copy = df_copy[df_copy['Sample Type'] == 'Rain']

    


    ##################################################################


    plt.figure(figsize=(10,6))

    plt.scatter(df_copy['Ca_ppm'], df_copy['Sr_ppm'],  alpha=0.7, s=70, c=df_copy['Longitude'], cmap='viridis')
    plt.xlabel('Ca')
    plt.ylabel('Sr')
    #plt.axline((0, 0), (1, 1), linewidth=1, color='b', label='1 HCO3 to 1 SO4')
    #plt.axline((0, 0), (2, 1), linewidth=1, color='r', label='2 HCO3 to 1 SO4')
    plt.colorbar(label='Longitude') 
    plt.legend()
    #convert axes to Logarithmic scale
    #plt.xscale("log")
    #plt.yscale("log")
    #for index, row in df_copy.iterrows():
    #    plt.text(row['Ca_ppm'], row['Sr_ppm'], row['Sample#'], fontsize=8, ha='center', va='bottom')
 
    #plt.xlim(0.1,100)
    #plt.ylim(0.1,100)

    plt.title('Ca vs Sr')
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
        scatter = ax.scatter(season_data['Na/Ca'], season_data['Sr/Ca'], alpha=0.7, s=70, c=season_data['Ca_ppm'], cmap='viridis')


    ax.set_ylabel('Sr/Ca')
    ax.set_xlabel('Na/Ca')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Ca_ppm')
    ax.set_title('Sr/Ca vs Na/Ca')
    
    ax.legend()
    plt.show()
    plt.close(fig)









def Investigation(df):
    
    # want to include only NICB valid samples. Takes only the unique code rows from dfneat if they also appear on unique_codes_valid
    
    df_copy = df.copy()
    
    df_copy = df_copy[df_copy['Sample type'] == 'Spring water']
    
    
    ## ADDING DEM DATA TO THE PLOT
    
    dem_path = '/Users/enrico/Desktop/Part III Project/DEM/AP_23217_PLR_F0550_RT1/AP_23217_PLR_F0550_RT1.dem.tif'
    
        
    # Define UTM bounds for Nepal DEM
    min_x, min_y, max_x, max_y = 346873.46875, 3078339.0, 362335.96875, 3115539.0

    
    # Define UTM bounds for Nepal DEM
    # Set CRS WKT strings
    utm_45n_wkt = 'PROJCS["WGS 84 / UTM zone 45N",GEOGCS["WGS 84",DATUM["WGS_1984",' \
                'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],' \
                'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],' \
                'UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]],' \
                'PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],' \
                'PARAMETER["central_meridian",87],PARAMETER["scale_factor",0.9996],' \
                'PARAMETER["false_easting",500000],PARAMETER["false_northing",0],' \
                'UNIT["metre",1,AUTHORITY["EPSG","9001"]],AUTHORITY["EPSG","32645"]]'

    wgs84_wkt = 'GEOGCS["WGS 84",DATUM["WGS_1984",' \
                'SPHEROID["WGS 84",6378137,298.257223563],' \
                'AUTHORITY["EPSG","6326"]],' \
                'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],' \
                'AUTHORITY["EPSG","4326"]]'

    # Open the DEM file and set CRS to UTM Zone 45N if undefined
    with rasterio.open(dem_path) as dem_raster:
        src_crs = rasterio.crs.CRS.from_wkt(utm_45n_wkt)  # Using WKT for UTM Zone 45N
        dst_crs = rasterio.crs.CRS.from_wkt(wgs84_wkt)    # Using WKT for WGS84

        # Create a Shapely box with UTM bounds
        dem_box = box(min_x, min_y, max_x, max_y)

        # Mask DEM data to these bounds
        out_image, out_transform = mask(dem_raster, [dem_box], crop=True)
        clipped_dem_array = out_image[0]

        # Calculate transform and reproject to WGS84
        dst_transform, width, height = calculate_default_transform(
            src_crs, dst_crs, clipped_dem_array.shape[1], clipped_dem_array.shape[0], *[min_x, min_y, max_x, max_y]
        )

        dst_array = np.zeros((height, width), dtype=np.float32)
        reproject(
            source=clipped_dem_array,
            destination=dst_array,
            src_transform=out_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )

        dst_array[dst_array < 0] = np.nan  # Filter out elevation values below 0

    # Define bounds in WGS84 (latitude/longitude) for plotting
    bounds = (
        dst_transform.c,  # min longitude
        dst_transform.f + dst_transform.e * height,  # min latitude
        dst_transform.c + dst_transform.a * width,  # max longitude
        dst_transform.f  # max latitude
    )

    # Generate the x and y coordinates in WGS84
    x = np.linspace(bounds[0], bounds[2], dst_array.shape[1])
    y = np.linspace(bounds[1], bounds[3], dst_array.shape[0])
    x, y = np.meshgrid(x, y)
    z = dst_array

    # Interpolator for DEM elevations
    interpolator = RegularGridInterpolator((y[:, 0], x[0, :]), z, bounds_error=False, fill_value=np.nan)

    # Ensure the DataFrame coordinates are in the same CRS as the DEM
    df_lons = df_copy['Longitude'].values
    df_lats = df_copy['Latitude'].values
    
    # Load the shapefile
    shapefile_path = "/Users/enrico/Desktop/Part III Project/DEM/Clipped_Shp_Melamchi.shp"
    gdf = gpd.read_file(shapefile_path)
    
    
    ##################################################################
       
   
    df_copy['Ca_mM'] = df_copy['Ca_ppm'] / 40.08
    df_copy['Sr_mM'] = df_copy['Sr_ppm'] / 87.62
    df_copy['Mg_mM'] = df_copy['Mg_ppm'] / 24.31
    df_copy['Si_mM'] = df_copy['Si_ppm'] / 28.09
    df_copy['Na_mM'] = df_copy['Na_ppm'] / 22.99
    
    df_copy['Ca/Na'] = df_copy['Ca_mM'] / df_copy['Na_mM']
    df_copy['Na/Ca'] = df_copy['Na_mM'] / df_copy['Ca_mM']
    df_copy['HCO3+CO32[mM]'] = df_copy['Alkalinity'] / 1000
    df_copy['HCO3/Na'] = df_copy['HCO3[mM]'] / df_copy['Na_mM']
    
    df_copy['Mg/Ca'] = df_copy['Mg_mM'] / df_copy['Ca_mM']
    df_copy['Mg/Na'] = df_copy['Mg_mM'] / df_copy['Na_mM']
    df_copy['Ca/Sr'] = df_copy['Ca_mM'] / df_copy['Sr_mM']
    df_copy['1000xSr/Ca'] = df_copy['Sr_mM'] / df_copy['Ca_mM'] * 1000
    
    df_copy['Si/Ca'] = df_copy['Si_mM'] / df_copy['Ca_mM']
    df_copy['Na/Ca'] = df_copy['Na_mM'] / df_copy['Ca_mM']
    
       
    ##################################################################
       
        
    # Exclude additional outliers
    outliers = ['Nep23-119', 'Nep23-110', 'NEP23-31', 'Nep23-133', 'NEP22-1', 'NEP23-01', 'Nep23-119']
    #df_copy = df_copy[~df_copy['Sample ID'].isin(outliers)]

    
    ##################################################################
    
    
    # Just accept traverse 3:
    df_copy = df_copy[df_copy['Traverse'] == 'Traverse 3']
    
    #Filter the latitude to be less than 27.9410 # taking in traverse 3a:
    df_copy = df_copy[(df_copy['Latitude'] < 27.9410) & (df_copy['Latitude'] > 27.9203)]
    
    
    
    
    
    
    # Create the figure and axes for side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    variable = 'Na/Ca'
    
    
    # Adjust the size of ax2 to 10:6
    ax2.set_box_aspect(6 / 10) 

    # Plot the DEM data in grayscale
    c = ax1.contourf(x, y, z, cmap='Greys', alpha=0.7)
    
    # Plot the shapefile
    gdf.plot(ax=ax1, facecolor='none', edgecolor='yellow', alpha=1, linewidth=1, label='Melamchi Watershed')

    # Define marker styles for the traverses
    traverse_markers = {
        'Traverse 1': 's',  # Square
        'Traverse 2': '^',  # Triangle
        'Traverse 3': '*',  # Star
        'Traverse 4': 'o'   # Circle
    }

    # Define color styles for the seasons
    season_colors = {
        'Nov_22': 'blue',
        'Apr_23': 'green',
        'Oct_23': 'red',
        'Sep_24': 'purple'
    }

    # Plot the points from df_copy, using the chosen variable for color and marker style for traverses
    for traverse, marker in traverse_markers.items():
        traverse_data = df_copy[df_copy['Traverse'] == traverse]
        for season, color in season_colors.items():
            season_data = traverse_data[traverse_data['Season'] == season]
            scatter = ax1.scatter(season_data['Longitude'], season_data['Latitude'], 
                                  c=color, s=70, alpha=0.7, edgecolor='k', 
                                  marker=marker, label=f'{traverse} - {season}')

    # Add a color legend for the seasons
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=season) 
               for season, color in season_colors.items()]
    ax1.legend(handles=handles, title='Season')



    # Set custom axis limits to match specified lat/long extents
    ax1.set_xlim(85.5, 85.6)
    ax1.set_ylim(27.82, 28.05)

    # Set labels and title
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title(f'DEM Map with {variable} Samples')

    ######### PLOT 2: Variable vs. Elevation #########

    # Colour the ax2 for traverse in the "Traverse" column, it will be Traverse 1, 2, 3 or 4
    traverse_colors = {
        'Traverse 1': 'blue',
        'Traverse 2': 'green',
        'Traverse 3': 'red',
        'Traverse 4': 'purple'
    }

    traverse_markers = {
        'Traverse 1': 's',  # Square
        'Traverse 2': '^',  # Triangle
        'Traverse 3': '*',  # Star
        'Traverse 4': 'o'   # Circle
    }
    
    for season, color in season_colors.items():
        season_data = df_copy[df_copy['Season'] == season]
        ax2.scatter(season_data['Elevation'], season_data[variable], alpha=0.7, s=70, color=color, marker='o', label=season)

    ax2.set_xlabel('Elevation')
    ax2.set_ylabel(f'{variable}')
    ax2.legend()
    plt.show()
    

    
    







    
df = pd.read_excel('Datasets/Nepal Master Sheet.xlsx')
    



Investigation(df)




#Altitude_Alkalinity_PDF(df)
#Altitude_Temperature(df)
#Altitude_pH(df)
#Altitude_TDS(df)
#Alkalinity_TDS(df)
#Ratios(df)  