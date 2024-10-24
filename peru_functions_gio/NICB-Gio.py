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
# - Before having our MasterScipt, this was used all in one line to run everything.
# - Undoubtedly tedious and since then unused


# def slice_relevant_data(df):
#     """Slice relevant data from the dataset"""
#     # select major elements for charge balance
#     elements = {'Ca', 'Mg', 'K', 'Na', 'HCO3', 'Cl', 'SO4', 'Al', 'As', 'Ba', 'Bi', 'Ce', 'Fe', 'Li', 'Rb', 'Sr', 'CO3', 'H2PO4'}

#     # loop through columns to retrieve columns with major elements in name with element+_in_mg_kg-1
#     # save as a list
#     major_elements = []
#     for element in elements:
#         col_standard = f"{element}_in_mg_kg-1"
#         col_alternate = f"{element}_in_mg_kg-1 [MA-0722]"
#         if col_standard in df.columns:
#             major_elements.append(col_standard)
#         elif col_alternate in df.columns:
#             major_elements.append(col_alternate)


#     # Add more columns from the original dataframe
#     additional_columns = ['longitude_new', 'longitude_old', 'latitude_new', 'latitude_old', 'calculated_discharge_in_m3_s-1', 'Alkalinity_CaCO3_in_mg_kg-1', 'altitude_in_m', 'water_body']  # Add more column names as needed
#     major_elements += additional_columns

#     # slice the dataframe to include only the major elements
#     df_slice = df[major_elements]

#     # append the sample ID to the sliced dataframe
#     df_slice.insert(0, 'unique_code', df['unique_code'])

#     #print(df_slice.columns)
    
#     #print(df_slice)

#     return df_slice


# def calculate_molar_concentration(df):
#     """Calculate molar concentration of the major elements"""
#     element_dict = {
#         'molecules': ['Ca', 'Mg', 'K', 'Na', 'HCO3', 'Cl', 'SO4', 'Al', 'As', 'Ba', 'Bi', 'Ce', 'Fe', 'Li', 'Rb', 'Sr', 'CO3', 'H2PO4'],
#         'molar_mass': [40.08, 24.31, 39.10, 22.99, 61.02, 35.45, 96.06, 26.98, 74.92, 137.33, 208.98, 140.12, 55.85, 6.94, 85.47, 87.62, 60.01, 97.99],
#         'valency': [2, 2, 1, 1, 1, 1, 2, 3, 3, 2, 3, 3, 2, 1, 1, 2, 2, 1]
#     }

#     # Suffixes
#     suffix = '_in_mg_kg-1'
#     alt_suffix = '_in_mg_kg-1 [MA-0722]'

#     df_copy = df.copy()

#     for element, molar_mass, valency in zip(element_dict['molecules'],
#                                             element_dict['molar_mass'],
#                                             element_dict['valency']):
#         # Check if the standard or alternate column exists
#         if element + suffix in df_copy.columns:
#             col_name = element + suffix
#         elif element + alt_suffix in df_copy.columns:
#             col_name = element + alt_suffix
#         else:
#             # Skip element if neither column exists
#             continue

#         # Create new molar columns
#         df_copy.loc[:, element + ' [aq] (mM)'] = df_copy[col_name] / molar_mass
#         df_copy.loc[:, element + ' [aq] (meq/L)'] = df_copy.loc[:, element + ' [aq] (mM)'] * valency

#     #print(df_copy.columns)

#     # Select relevant columns
#     df_copy = df_copy[['unique_code'] + [col for col in df_copy.columns if ' [aq] (mM)' in col] + [col for col in df_copy.columns if ' [aq] (meq/L)' in col]+ ['latitude_old'] + ['latitude_new'] + ['longitude_old'] + ['longitude_new'] + ['calculated_discharge_in_m3_s-1'] + ['Alkalinity_CaCO3_in_mg_kg-1'] + ['altitude_in_m'] + ['water_body']]

#     #print(df_copy)

#     return df_copy


# def molar_conc_seawater(df):
#     """Calculate molar concentration of the major elements"""
#     element_dict = {
#         'molecules': ['Ca', 'Mg', 'K', 'Na', 'HCO3', 'Cl', 'SO4', 'Al', 'As', 'Ba', 'Bi', 'Ce', 'Fe', 'Li', 'Rb', 'Sr', 'CO3', 'H2PO4'],
#         'molar_mass': [40.08, 24.31, 39.10, 22.99, 61.02, 35.45, 96.06, 26.98, 74.92, 137.33, 208.98, 140.12, 55.85, 6.94, 85.47, 87.62, 60.01, 97.99],
#         'valency': [2, 2, 1, 1, 1, 1, 2, 3, 3, 2, 3, 3, 2, 1, 1, 2, 2, 1]
#     }

#     # Suffixes
#     suffix = ''
#     alt_suffix = ''

#     df_copy = df.copy()

#     for element, molar_mass, valency in zip(element_dict['molecules'],
#                                             element_dict['molar_mass'],
#                                             element_dict['valency']):
#         # Check if the standard or alternate column exists
#         if element + suffix in df_copy.columns:
#             col_name = element + suffix
#         elif element + alt_suffix in df_copy.columns:
#             col_name = element + alt_suffix
#         else:
#             # Skip element if neither column exists
#             continue

#         # Create new molar columns
#         df_copy.loc[:, element + ' [aq] (mM)'] = df_copy[col_name] / molar_mass
#         df_copy.loc[:, element + ' [aq] (meq/L)'] = df_copy.loc[:, element + ' [aq] (mM)'] * valency

#     #print(df_copy.columns)

#     # Select relevant columns
#     #df_copy = [ [col for col in df_copy.columns if ' [aq] (mM)' in col] + [col for col in df_copy.columns if ' [aq] (meq/L)' in col] ]

#     #print(df_copy)

#     return df_copy


# def charge_balance(df):
#     """Function to charge balance"""
#     cations = ['Ca', 'Mg', 'K', 'Na', 'Al', 'As', 'Ba', 'Bi', 'Fe', 'Li', 'Rb', 'Sr']
#     anions = ['HCO3', 'Cl', 'SO4', 'H2PO4', 'CO3']

#     # Calculate the sum of cations and anions
#     df['sum_cations'] = df[[col for col in df.columns if any(cation in col for cation in cations) and ' [aq] (meq/L)' in col]].sum(axis=1)

#     df['sum_anions'] = df[[col for col in df.columns if any(anion in col for anion in anions) and ' [aq] (meq/L)' in col]].sum(axis=1)

#     # NICB calculation
#     df['NICB diff'] = (df['sum_cations'] - df['sum_anions'])
#     df['NICB sum'] = (df['sum_cations'] + df['sum_anions'])

#     # Remove rows with NICB sum = 0
#     df = df[df['NICB sum'] != 0]

#     df['NICB'] = df['NICB diff'] / df['NICB sum']

#     return df


# def plot_NICB(df):
#     """Plot NICB distribution"""
#     # Plot the NICB distribution as a histogram
#     fig, ax = plt.subplots()

#     # Plot KDE
#     (df['NICB']*100).plot.kde(ax=ax, legend=False, title='NICB distribution (%)')

#     plt.axvline(x = 10, color = 'b', label = '10% Bounds')
#     plt.axvline(x = -10, color = 'b')

#     # Save plot
#     plt.savefig('chemweathering/figures/NICB_distribution_percent.png')
#     plt.close(fig)


# def convert_latitude(row):
#     if pd.notna(row['latitude_new']):
#         return row['latitude_new']
#     elif pd.notna(row['latitude_old']):
#         match = re.match(r'(\d+)\s+(\d+\.\d+)', str(row['latitude_old']))
#         if match:
#             degrees = int(match.group(1))
#             minutes = float(match.group(2))
#             decimal_degrees = -(degrees + (minutes / 60))
#             return decimal_degrees
#     return None


# def convert_longitude(row):
#     # Function to convert degrees and minutes to decimal degrees for longitude

#     if pd.notna(row['longitude_new']):
#         return row['longitude_new']
#     elif pd.notna(row['longitude_old']):
#         match = re.match(r'(\d+)\s+(\d+\.\d+)', str(row['longitude_old']))
#         if match:
#             degrees = int(match.group(1))
#             minutes = float(match.group(2))
#             decimal_degrees = -(degrees + (minutes / 60))
#             return decimal_degrees
#     return None


# def GDPlot(df, df2):

#     # want to include only NICB valid samples. Takes only the unique code rows from dfneat if they also appear on unique_codes_valid
    
    
#     df = df[df['unique_code'].isin(df2['unique_code_valid'])]
    
#     #### Might I just be able to do df instead of dfnew??
#     #print(df.columns)

#     df['latitude_converted'] = df.apply(convert_latitude, axis=1)
#     df['longitude_converted'] = df.apply(convert_longitude, axis=1)

# # Drop rows with NaN values in converted latitude and longitude
#     df.dropna(subset=['longitude_converted', 'latitude_converted'], inplace=True)

#     plt.figure(figsize=(10,6))

#     df['HCO3Na'] = df['HCO3 [aq] (mM)']/df['Na [aq] (mM)']
#     df['CaNa'] = df['Ca [aq] (mM)']/df['Na [aq] (mM)']

#     plt.scatter(df['HCO3 [aq] (mM)'], df['SO4 [aq] (mM)'], alpha=0.7, s=70, c=df['longitude_converted'], cmap='viridis')
#     plt.xlabel('HCO3 [aq] (mM)')
#     plt.ylabel('SO4 [aq] (mM)')
#     plt.axline((0, 0), (1, 1), linewidth=1, color='b', label='1 HCO3 to 1 SO4')
#     plt.axline((0, 0), (2, 1), linewidth=1, color='r', label='2 HCO3 to 1 SO4')
#     plt.colorbar(label='Longitude') 
#     plt.legend()
#     #convert axes to Logarithmic scale
#     #plt.xscale("log")
#     #plt.yscale("log")

#     plt.title('HCO3/SO4 Plot')
#     plt.savefig('chemweathering/figures/HCO3SO4.png')
#     plt.close()
#     #plt.show()


# def haversine_distance(lat1, lon1, lat2, lon2):
#     """
#     Calculate the Haversine distance between two sets of lat/lon coordinates.
#     """
#     # Radius of the Earth in kilometers
#     R = 6371.0

#     # Convert latitude and longitude from degrees to radians
#     lat1_rad = math.radians(float(lat1))
#     lon1_rad = math.radians(float(lon1))
#     lat2_rad = math.radians(float(lat2))
#     lon2_rad = math.radians(float(lon2))

#     # Compute differences in coordinates
#     dlat = lat2_rad - lat1_rad
#     dlon = lon2_rad - lon1_rad

#     # Haversine formula
#     a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

#     # Distance in kilometers
#     distance = R * c
#     return distance


# def DistPlot(df):
#     fixed_lat = -13.32468333
#     fixed_lon = -76.52861944

#     # Apply the haversine_distance function to each row in the DataFrame
#     df_copy = df.copy()
#     df_copy['Distance (km)'] = df_copy.apply(
#         lambda row: haversine_distance(fixed_lat, fixed_lon, row['latitude_converted'], row['longitude_converted']),
#         axis=1
#     )

#     ## Nice use of lambda
    
#     #print(df_copy)


#     max_value_row = df_copy.loc[df_copy['*K [aq] (mM)'].idxmax()]
#     #print(df_copy['*K [aq] (mM)'].idxmax())
#     #print(max_value_row['unique_code']) 
    
#     ## Want to remove row with RC15c-44748
    
#     df_copy.to_csv('chemweathering/data/CorrectedValues.csv', index=False)
    
#     df_copy = df_copy[~df_copy['unique_code'].str.contains('T1DW-0322', na=False)]
    
    

    
#     for column in df_copy.columns:
#         if '*' in column:
#             plt.figure(figsize=(10,6))
#             scatter = plt.scatter(df_copy['Distance (km)'], df_copy[column], alpha=0.7, s=70, c=df_copy['altitude_in_m'], cmap='viridis', label=column)
#             plt.xlabel('Distance (km) from Shore')
#             plt.ylabel(column)
#             plt.colorbar(label='Altitude (m)')
#             plt.title(f'Scatter plot of {column} vs. Distance (km)')

#             # Annotate each point with unique_code
#             #for index, row in df_copy.iterrows():
#             #    plt.text(row['Distance (km)'], row[column], row['unique_code'], fontsize=8, ha='center', va='bottom')

#             plt.legend()  # Include legend with labels
#             plt.savefig('chemweathering/figures/' + column.replace('*', '').strip() + '.png') 
#             #plt.show()  # Show each plot individually
#             plt.close()
            

# def mixdiag(df):
    
#     df_copy = df.copy()
    
#     df_copy = df_copy[~df_copy['unique_code'].str.contains('T1DW-0322', na=False)]
    
#     #########
    
#     # want to differentiate between River and tributary and spring
#     #########
    
#     ### print(df_copy.sort_values(by='altitude_in_m', ascending=False))
#     ### Just checking highest values

    
#    # Ensure all water_body values are mapped to valid markers
#     markers = {'mainstream': 'o', 'tributary': '^', 'spring': '*'}

#     # Create a figure and axis object
#     fig, ax = plt.subplots(figsize=(10, 6))


#     df_copy['MgNa'] = df_copy['*Mg [aq] (mM)'] / df_copy['*Na [aq] (mM)']
#     df_copy['CaNa'] = df_copy['*Ca [aq] (mM)'] / df_copy['*Na [aq] (mM)']

#     for water_body, marker in markers.items():
#         subset = df_copy[df_copy['water_body'] == water_body]
#         ax.scatter(subset['*Ca [aq] (mM)'] / subset['*Na [aq] (mM)'],
#                subset['*Mg [aq] (mM)'] / subset['*Na [aq] (mM)'],
#                s=70, alpha=0.7,
#                c=subset['altitude_in_m'],
#                cmap='viridis',
#                marker=marker)

#     ax.set_xlabel('Ca/Na molar ratio')
#     ax.set_ylabel('Mg/Na molar ratio')
#     ax.set_title('Mg/Na against Ca/Na')
#     ax.set_xscale("log")
#     ax.set_yscale("log")
    
#     # Add colorbar separately
#     cb = fig.colorbar(ax.collections[0], ax=ax, label='Altitude (m)')

#     # Create legend with custom markers and labels
#     for water_body, marker in markers.items():
#         ax.scatter([], [], marker=marker, color='black', label=water_body)

#     ax.legend(scatterpoints=1, labelspacing=1, title='Water Body', loc='upper left')
    
#     #plt.legend()  # Include legend with labels
#     plt.savefig('chemweathering/figures/MgNa-CaNa.png')     
#     plt.close(fig)
    
#     #plt.show()
    
#      # Annotate each point with unique_code
#     #for index, row in df_copy.iterrows():
#     #    plt.text(row['Distance (km)'], row[column], row['unique_code'], fontsize=8, ha='center', va='bottom')
    
    
#     ########## NEW PLOT
    
    
    
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     df_copy['HCO3Na'] = df_copy['*HCO3 [aq] (mM)'] / df_copy['*Na [aq] (mM)']
    
#     for water_body, marker in markers.items():
#         subset = df_copy[df_copy['water_body'] == water_body]
#         ax.scatter(subset['*Ca [aq] (mM)'] / subset['*Na [aq] (mM)'],
#                subset['*HCO3 [aq] (mM)'] / subset['*Na [aq] (mM)'],
#                s=70, alpha=0.7,
#                c=subset['altitude_in_m'],
#                cmap='viridis',
#                marker=marker)

#     ax.set_xlabel('Ca/Na molar ratio')
#     ax.set_ylabel('HCO3/Na molar ratio')
#     ax.set_title('HCO3/Na against Ca/Na')
#     ax.set_xscale("log")
#     ax.set_yscale("log")
    
#     # Add colorbar separately
#     cb = fig.colorbar(ax.collections[0], ax=ax, label='Altitude (m)')

#     # Create legend with custom markers and labels
#     for water_body, marker in markers.items():
#         ax.scatter([], [], marker=marker, color='black', label=water_body)

#     ax.legend(scatterpoints=1, labelspacing=1, title='Water Body', loc='upper left')
    
#     plt.savefig('chemweathering/figures/HCO3Na-CaNa.png') 
#     plt.close(fig)
#     #plt.show()  # Show each plot individually


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


# def map_with_log_colorscale(df):
    
#     ###### Below Potentially removable if you merge the datasets with GDPlot 
#     # Create a Folium map centered around the mean of latitude and longitude
    
#     df['latitude_converted'] = df.apply(convert_latitude, axis=1)
#     df['longitude_converted'] = df.apply(convert_longitude, axis=1)

# # Drop rows with NaN values in converted latitude and longitude


#     df.dropna(subset=['longitude_converted', 'latitude_converted'], inplace=True)
    
#     df['CaNa'] = df['Ca [aq] (mM)']/df['Na [aq] (mM)']

#     ###### Above Potentially removable if you merge the datasets with GDPlot
    
    
    
    
    
    
    
#     m = folium.Map(location=[df.latitude_converted.mean(), df.longitude_converted.mean()], zoom_start=10)

#     # Add Esri Satellite tile layer
#     tile = folium.TileLayer(
#         tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
#         attr='Esri',
#         name='Esri Satellite',
#         overlay=False,
#         control=True
#     ).add_to(m)

#     # Define logarithmic color scale using matplotlib
#     cmap = get_cmap('viridis')  # Choose colormap, e.g., Viridis
#     norm = LogNorm(vmin=0.1, vmax=100)  # Adjust vmin and vmax based on your data range and logarithmic scale

#     # Iterate through the dataframe and add CircleMarkers to the map
#     for index, location_info in df.iterrows():
#         value = location_info['CaNa']
        
#         # Ensure the value is within the range of the colormap
#         if 0.1 <= value <= 100:
#             color = cmap(norm(value))  # Get color from colormap based on logarithmic scale
#             color_hex = plt.cm.colors.rgb2hex(color)
            
#             folium.CircleMarker(
#                 [location_info["latitude_converted"], location_info["longitude_converted"]],
#                 popup=f"Ca/Na: {location_info['CaNa']}<br><br>Sample: {location_info['unique_code']}",
#                 radius=10,
#                 color="black",
#                 weight=1,
#                 fill_opacity=0.8,
#                 opacity=1,
#                 fill_color=color_hex,
#                 fill=True
#             ).add_to(m)

#     # Legend HTML
#        # Legend HTML
#     legend_html = '''
#     <div style="position: fixed; 
#                 top: 10px; right: 10px; width: 280px; height: 100px; 
#                 border: 2px solid grey; z-index: 9999; font-size: 14px;
#                 background-color: white; opacity: 0.9; padding: 10px;">
#     <strong>Logarithmic Colormap, Ca/Na</strong><br>
#     <text x="10" y="50" text-anchor="start" fill="black">&emsp;&emsp;0.1&emsp;&emsp;&emsp; 1&emsp;&emsp;&emsp;&emsp; 10&emsp;&emsp;&emsp; 100</text>
#     <svg height="30" width="250">
#       <rect x="0" y="0" width="250" height="30" style="fill:rgb(255,255,255);stroke-width:1;stroke:rgb(0,0,0)" />
#       <rect x="5" y="5" width="50" height="20" style="fill:{color_min}" />
#       <rect x="65" y="5" width="50" height="20" style="fill:{color_mid1}" />
#       <rect x="125" y="5" width="50" height="20" style="fill:{color_mid2}" />
#       <rect x="185" y="5" width="50" height="20" style="fill:{color_max}" />
#     </svg>
#     </div>
#     '''.format(color_min=plt.cm.colors.rgb2hex(cmap(norm(0.1))),
#                color_mid1=plt.cm.colors.rgb2hex(cmap(norm(1))),
#                color_mid2=plt.cm.colors.rgb2hex(cmap(norm(10))),
#                color_max=plt.cm.colors.rgb2hex(cmap(norm(100))))
    
#     m.get_root().html.add_child(folium.Element(legend_html))


#     # Save the map to the specified folder
#     folder_path = 'chemweathering/data/figures'
#     file_name = 'map_logcana_gio.html'
#     os.makedirs(folder_path, exist_ok=True)
#     full_path = os.path.join(folder_path, file_name)
#     m.save(full_path)


# ## Just be aware that currently GDPlot and map_with_log_colorscale operate under two different dataframes because the GDPlot code looks to use a NICB filtered dataset 
# ## and the log colorscale one not yet. As such they both use the convert lat and long functions because of how the code is set up. The removable bits are demarcated 
# ## in the map def.

# #### Update should have fixed this


# def NICB_Valid(df_neat):
#     unique_codes_valid = []
#     unique_codes_invalid = []
#     nicb_valid = []
#     nicb_invalid = []

#     #Iterate through NICB values
#     for index, i in df_neat['NICB'].items():
#         unique_code = df_neat.loc[index, 'unique_code']
#         if i > 0.1 or i < -0.1:
#             nicb_invalid.append(i)
#             unique_codes_invalid.append(unique_code)
#         else:
#             nicb_valid.append(i)
#             unique_codes_valid.append(unique_code)

#     # Calculate the maximum length to pad with None
#     max_length = max(len(nicb_valid), len(nicb_invalid))

#     # Pad lists with None to match the maximum length
#     nicb_valid += [None] * (max_length - len(nicb_valid))
#     nicb_invalid += [None] * (max_length - len(nicb_invalid))
#     unique_codes_valid += [None] * (max_length - len(unique_codes_valid))
#     unique_codes_invalid += [None] * (max_length - len(unique_codes_invalid))

#     # Create the new DataFrame
#     NICB_Balance = pd.DataFrame({
#         'unique_code_valid': unique_codes_valid,
#         'NICB_Valid': nicb_valid,
#         'unique_code_invalid': unique_codes_invalid,
#         'NICB_Invalid': nicb_invalid
#     })  
    
#     #print(len(NICB_Balance['unique_code_valid']))
            
#     NICB_Balance.to_csv('chemweathering/data/ValidNICB.csv', index=False)

#     return (NICB_Balance)



# def Chloride_Correction(df, df2):
#     # Seawater data in typical concentrations in ppm
#     seawater_data = {
#         'Parameter': ['Cl', 'Na', 'SO4', 'Mg', 'Ca', 'K', 'HCO3', 'Sr'],
#         'Typical Seawater': [19353, 10693, 2712, 1284, 412, 399, 126, 13]
#     }
    
#     #ppm (Pretet et al 2014)

#     # Convert to DataFrame
#     rain = pd.DataFrame(seawater_data).set_index('Parameter').transpose()

#     # Convert concentrations to molar units (assuming molar_conc_seawater function is defined)
#     rain_processed = molar_conc_seawater(rain)

#     # Ensure columns are in correct order
#     rain_processed = rain_processed.loc[:, ['Ca [aq] (mM)', 'Mg [aq] (mM)', 'K [aq] (mM)', 'Na [aq] (mM)', 'HCO3 [aq] (mM)', 'Cl [aq] (mM)', 'SO4 [aq] (mM)', 'Sr [aq] (mM)']]
    
#     #print(rain_processed)
    
#     # Filter df to include only valid samples
#     df = df[df['unique_code'].isin(df2['unique_code_valid'])]

#     # Create a copy of df for modifications
#     df_copy = df.copy()

#     # Correction calculation
    
#     # Iterate over each element (column) in the rain_processed DataFrame
#     for element in rain_processed.columns:
#     # Check if the current element exists as a column in the df_copy DataFrame
    
#         if element in df_copy.columns:
#         # Get the value of the current element from the rain_processed DataFrame (assuming there's only one value)
        
#             rain_element_value = rain_processed[element].values[0]
        
#         # Get the value of 'Cl [aq] (mM)' from the rain_processed DataFrame (assuming there's only one value)
        
#             cl_rain_value = rain_processed['Cl [aq] (mM)'].values[0]
        
#         # Check to ensure the Cl value is not zero to avoid division by zero
#             if cl_rain_value != 0:
                
#             # Perform the correction calculation:
#             # For each row in df_copy, subtract the scaled rain element value from the river element value
            
#                 df_copy['*' + element] = df_copy[element] - (rain_element_value / cl_rain_value) * df_copy['Cl [aq] (mM)']
#             else:
#             # Print a warning message if the Cl value is zero to avoid division by zero
#                 print(f"Division by zero encountered for element: {element}")
#         else:
#         # Print a message if the current element is not found in the df_copy DataFrame
#             print(f"Element {element} not found in df_copy")

#     # Select relevant columns for final output
#     df_copy = df_copy[['unique_code'] + [col for col in df_copy.columns if '*' in col] + ['latitude_converted', 'longitude_converted', 'NICB', 'calculated_discharge_in_m3_s-1', 'Alkalinity_CaCO3_in_mg_kg-1', 'altitude_in_m', 'water_body']]

#     return df_copy



#     #print(df_copy.columns)
    
    
    
    
    
    
    
    
    
#     #print(df)
    
#     #print(rain)



# def Chloride_Correction2(df, df2):
#     # Seawater data in typical concentrations in ppm
#     seawater_data = {
#         'Parameter': ['Cl', 'Na', 'SO4', 'Mg', 'Ca', 'K', 'HCO3', 'Sr'],
#         'Typical Seawater': [19353, 10693, 2712, 1284, 412, 399, 126, 13]
#     }
    
#     #ppm (Pretet et al 2014)

#     # Convert to DataFrame
#     rain = pd.DataFrame(seawater_data).set_index('Parameter').transpose()

#     # Convert concentrations to molar units (assuming molar_conc_seawater function is defined)
#     rain_processed = molar_conc_seawater(rain)

#     # Ensure columns are in correct order
#     rain_processed = rain_processed.loc[:, ['Ca [aq] (mM)', 'Mg [aq] (mM)', 'K [aq] (mM)', 'Na [aq] (mM)', 'HCO3 [aq] (mM)', 'Cl [aq] (mM)', 'SO4 [aq] (mM)', 'Sr [aq] (mM)']]
    
#     #print(rain_processed)
    
#     # Filter df to include only valid samples
#     df = df[df['unique_code'].isin(df2['unique_code_valid'])]

#     # Create a copy of df for modifications
#     df_copy = df.copy()
    
#     #cl_minimum = the lowest cl value in df_copy:
#     cl_minimum = df_copy['Cl [aq] (mM)'].min()
    

#     # Correction calculation
    
#     # Iterate over each element (column) in the rain_processed DataFrame
#     for element in rain_processed.columns:
#     # Check if the current element exists as a column in the df_copy DataFrame
    
#         if element in df_copy.columns:
#         # Get the value of the current element from the rain_processed DataFrame (assuming there's only one value)
        
#             rain_element_value = rain_processed[element].values[0]
        
        
#         # Get the value of 'Cl [aq] (mM)' from the rain_processed DataFrame (assuming there's only one value)
        
#             cl_rain_value = rain_processed['Cl [aq] (mM)'].values[0]
            
#             cl_minimum_star = cl_minimum - cl_rain_value
            
#             print('Cl minimum star:',
#                 cl_minimum_star)  
            
            
            
        
#         # Check to ensure the Cl value is not zero to avoid division by zero
#             if cl_rain_value != 0:
#                 if cl_minimum_star >= 0:
#                     # Perform the correction calculation:
#                     # For each row in df_copy, subtract the scaled rain element value from the river element value
#                     df_copy['*' + element] = df_copy[element] - ((rain_element_value / cl_rain_value) * (df_copy['Cl [aq] (mM)'] - cl_minimum_star))
#                 else:
#                     # Set cl_minimum_star to zero
#                     cl_minimum_star = 0
#                     # Perform the correction calculation with cl_minimum_star set to zero
#                     df_copy['*' + element] = df_copy[element] - ((rain_element_value / cl_rain_value) * (df_copy['Cl [aq] (mM)'] - cl_minimum_star))
            

#         else:
#         # Print a message if the current element is not found in the df_copy DataFrame
#             print(f"Element {element} not found in df_copy")
            
#     ## make a plot of the Cl* values in df_copy:
#     plt.figure(figsize=(10,6))
#     plt.scatter(df_copy['altitude_in_m'], df_copy['*Cl [aq] (mM)'], alpha=0.7, s=70)
#     plt.ylabel('*Cl [aq] (mM)')
#     plt.xlabel('Altitude (m)')
#     plt.title('Scatter plot of *Cl vs. Altitude')
#     plt.savefig('chemweathering/figures/Clstar.png')
#     plt.show()
#     plt.close()
            

#     # Select relevant columns for final output
#     df_copy = df_copy[['unique_code'] + [col for col in df_copy.columns if '*' in col] + ['latitude_converted', 'longitude_converted', 'NICB', 'calculated_discharge_in_m3_s-1', 'Alkalinity_CaCO3_in_mg_kg-1', 'altitude_in_m', 'water_body']]

#     return df_copy



# def Piper(df):
    
#     df_copy = df.copy()
    
#     #print(df_copy)
    
    
#     #we are doing uncorrected for now
    
    
    
    
    
#     # Drop rows with NaN values in discharge
#     df_copy.dropna(subset=['calculated_discharge_in_m3_s-1'], inplace=True)
    
    
#     # Ensure all values in 'calculated_discharge_in_m3_s-1' are numeric
#     df_copy['calculated_discharge_in_m3_s-1'] = pd.to_numeric(df_copy['calculated_discharge_in_m3_s-1'], errors='coerce')
    
#     # Drop any rows that could not be converted to numeric values
#     df_copy.dropna(subset=['calculated_discharge_in_m3_s-1'], inplace=True)
    
    
#     ####### now do the same but for alkalinity
    
#     # Drop rows with NaN values in alkalinity
#     df_copy.dropna(subset=['Alkalinity_CaCO3_in_mg_kg-1'], inplace=True)
    
    
#     # Ensure all values in alkalinity are numeric
#     df_copy['Alkalinity_CaCO3_in_mg_kg-1'] = pd.to_numeric(df_copy['Alkalinity_CaCO3_in_mg_kg-1'], errors='coerce')
    
#     # Drop any rows that could not be converted to numeric values
#     df_copy.dropna(subset=['Alkalinity_CaCO3_in_mg_kg-1'], inplace=True)

    
#     # Convert the size column to a list or else it gives an invalid error
#     #size_list = df_copy['calculated_discharge_in_m3_s-1'].tolist()
    
#     #those values are too small to work with
    

    
#     # Apply a logarithmic transformation to the size column
#     log_size = np.log(df_copy['calculated_discharge_in_m3_s-1'] + 1)

    
#     df_copy['Na+K'] = df_copy['Na [aq] (mM)'] + df_copy['K [aq] (mM)']
    
#     fig1 = px.scatter_ternary(df_copy, a="Mg [aq] (mM)", b="Ca [aq] (mM)", c="Na+K", hover_name="unique_code", color="water_body", size='altitude_in_m', size_max=15,
#                              color_discrete_map = {"mainstream": "blue", "tributary": "green", "spring":"red"})

#     #fig1.show()
    
#     df_copy['Na+K (mM)'] = df_copy['Na [aq] (mM)'] + df_copy['K [aq] (mM)']
    
#     df_copy['CO3+HCO3 (mM)'] = df_copy['Alkalinity_CaCO3_in_mg_kg-1'] / 100.0869
    
#     fig2 = px.scatter_ternary(df_copy, a="CO3+HCO3 (mM)", b="Cl [aq] (mM)", c="SO4 [aq] (mM)", hover_name="unique_code", color="water_body", size='altitude_in_m', size_max=15,
#                              color_discrete_map = {"mainstream": "blue", "tributary": "green", "spring":"red"})

#     #fig2.show()
    
    
#     #### 
#     ### Now plotting SO4 and Na+K
    
#     df_copy['Na+K Normal (mM)'] = df_copy['Na+K (mM)'] / (df_copy['K [aq] (mM)'] + df_copy['Na [aq] (mM)'] + df_copy['Ca [aq] (mM)'] + df_copy['Mg [aq] (mM)'])
    
#     df_copy['SO4 Normal (mM)'] = df_copy['SO4 [aq] (mM)'] / (df_copy['CO3+HCO3 (mM)'] + df_copy['SO4 [aq] (mM)'] + df_copy['Cl [aq] (mM)'])
    
    
    
#     plt.figure(figsize=(10,6))

#     plt.scatter(df_copy['Na+K Normal (mM)'], df_copy['SO4 Normal (mM)'], alpha=0.7, s=70, c=df_copy['altitude_in_m'], cmap='viridis')
#     plt.xlabel('Na+K Normal (mM)')
#     plt.ylabel('SO4 Normal (mM)')
#     #plt.axline((0, 0), (1, 1), linewidth=1, color='b', label='1 HCO3 to 1 SO4')
#     #plt.axline((0, 0), (2, 1), linewidth=1, color='r', label='2 HCO3 to 1 SO4')
#     plt.colorbar(label='Altitude (m)') 
#     #plt.legend()
#     #convert axes to Logarithmic scale
#     #plt.xscale("log")
#     #plt.yscale("log")

#     plt.title('Normalised Na+K and SO4 (Molar)')
#     plt.savefig('chemweathering/figures/Molar-Prop.png')
#     #plt.show()
#     plt.close()
    
#     return(df_copy)
    


# def RockData():
    
#     df_rock = pd.read_excel('chemweathering/data/Mock-Data.xlsx')
    
#     df_rock = df_rock[df_rock['Country'] == 'Peru']
    
#     df_rock = df_rock[df_rock['Sample_type'].str.contains('metamorphic')==False]  # drop metamorphic
    
#     df_rock.dropna(subset=['CaO', 'MgO', 'Na2O'], inplace=True)
    
#     def compute_molar_concentration(df, element, oxide, molar_masses):
#         df[f'{element} (mol/100g)'] = df[oxide] / (molar_masses[element] + molar_masses['O'])
#         return df
    
#     molar_masses = {
#         'Ca': 40.078,
#         'Mg': 24.305,
#         'Na': 22.990,
#         'O': 15.999
#     }

#     # Define a function to calculate the ratios and log values
#     def compute_ratios_and_logs(df):
#         df['CaNa'] = df['Ca (mol/100g)'] / df['Na (mol/100g)']
#         df['MgNa'] = df['Mg (mol/100g)'] / df['Na (mol/100g)']
#         df['LogCaNa'] = np.log(df['CaNa'])
#         df['LogMgNa'] = np.log(df['MgNa'])
#         return df

#     # Define a function to compute the mean and standard deviation
#     def compute_stats(df):
#         mean_log_ca_na = df['LogCaNa'].mean()
#         std_log_ca_na = df['LogCaNa'].std()
#         mean_log_mg_na = df['LogMgNa'].mean()
#         std_log_mg_na = df['LogMgNa'].std()
    
#         #print(f'Mean Log Ca/Na is: {mean_log_ca_na}')
#         #print(f'Std Dev Log Ca/Na is: {std_log_ca_na}')
#         #print(f'Mean Log Mg/Na is: {mean_log_mg_na}')
#         #print(f'Std Dev Log Mg/Na is: {std_log_mg_na}')
    
#         return mean_log_ca_na, std_log_ca_na, mean_log_mg_na, std_log_mg_na

#     # List of elements and their corresponding oxides
#     elements_oxides = {
#         'Ca': 'CaO',
#         'Mg': 'MgO',
#         'Na': 'Na2O'
#     }

#     # Calculate molar concentrations for each element
#     for element, oxide in elements_oxides.items():
#         if element == 'Na':  # Special case for Na2O to account for two Na atoms
#             df_rock[f'{element} (mol/100g)'] = (df_rock[oxide] / (2 * molar_masses[element] + molar_masses['O'])) * 2
#         else:
#             df_rock = compute_molar_concentration(df_rock, element, oxide, molar_masses)

#     # Apply the ratio and log calculations
#     df_rock = compute_ratios_and_logs(df_rock)

#     # Compute and print the statistics
#     mean_log_ca_na, std_log_ca_na, mean_log_mg_na, std_log_mg_na = compute_stats(df_rock)
    
    

    
    
    
    
#     #########
#     ######### Maximum Likelihood Estimation for log-normal distribution parameters

#     log_mg_na = df_rock['LogMgNa'].replace([np.inf, -np.inf], np.nan).dropna()
#     log_ca_na = df_rock['LogCaNa'].replace([np.inf, -np.inf], np.nan).dropna()
    

#     mu_mle_mg, sigma_mle_mg = norm.fit(log_mg_na)
#     mu_mle_ca, sigma_mle_ca = norm.fit(log_ca_na)
    
#     processed_mean_mg_na = math.exp(mu_mle_mg + 0.5*((sigma_mle_mg)**2))
    
#     processed_sigma_mg_na = np.sqrt((math.exp(sigma_mle_mg**2) -1)*(math.exp(2*mu_mle_mg + (sigma_mle_mg**2))))
    
#     processed_mean_ca_na = math.exp(mu_mle_ca + 0.5*((sigma_mle_ca)**2))
    
#     processed_sigma_ca_na = np.sqrt((math.exp(sigma_mle_ca**2) -1)*(math.exp(2*mu_mle_ca + (sigma_mle_ca**2))))
    
    
    
#     #print(f'MLE estimated μ for Log Mg/Na: {mu_mle_mg}')
#     #print(f'MLE estimated σ for Log Mg/Na: {sigma_mle_mg}')
#     #print(f'Estimated μ for Mg/Na: {processed_mean_mg_na}')
#     #print(f'Estimated σ for Mg/Na: {processed_sigma_mg_na}')
    
    
        
#     #print(f'MLE estimated μ for Log Ca/Na: {mu_mle_ca}')
#     #print(f'MLE estimated σ for Log Ca/Na: {sigma_mle_ca}')
#     #print(f'Estimated μ for Ca/Na: {processed_mean_ca_na}')
#     #print(f'Estimated σ for Ca/Na: {processed_sigma_ca_na}')
    
    
    
    
    
    
    
#     #########
#     ######### Plot histogram and fitted distribution
    
#     plt.figure(figsize=(10,6))
#     plt.hist(df_rock['MgNa'], bins=100, density=True, alpha=0.6, color='g')
    
#     xmin, xmax = plt.xlim()
#     x = np.linspace(xmin, xmax, 100)
    
#     #This creates 100 equally spaced values (x) between xmin and xmax. These values are used to generate the points on the fitted log-normal distribution curve.
    
#     p = norm.pdf(np.log(x), mu_mle_mg, sigma_mle_mg)
    
#     # np.log(x) takes the natural logarithm of each value in x because the data is assumed to follow a log-normal distribution.
#     # norm.pdf() computes the probability density function (PDF) of the normal distribution for these log-transformed x values, 
#     # using the MLE-estimated parameters mu_mle and sigma_mle.
    
#     plt.plot(x, p, 'k', linewidth=2)
    
#     #plotting the fitted distribution
    
#     plt.title('Mg/Na Histogram and Fitted Log-Normal Distribution')
#     plt.xlabel('Mg/Na')
#     plt.ylabel('Density')
#     plt.savefig('chemweathering/figures/HistogramMgNa.png')
#     plt.close()
#     #plt.show()
    
#     ######
#     ######
    
    
#     # Plot histogram and fitted distribution
#     plt.figure(figsize=(10,6))
#     plt.hist(df_rock['CaNa'], bins=100, density=True, alpha=0.6, color='g')
    
#     xmin, xmax = plt.xlim()
#     x = np.linspace(xmin, xmax, 100)
    
#     #This creates 100 equally spaced values (x) between xmin and xmax. These values are used to generate the points on the fitted log-normal distribution curve.
    
#     p = norm.pdf(np.log(x), mu_mle_ca, sigma_mle_ca)
    
#     # np.log(x) takes the natural logarithm of each value in x because the data is assumed to follow a log-normal distribution.
#     # norm.pdf() computes the probability density function (PDF) of the normal distribution for these log-transformed x values, 
#     # using the MLE-estimated parameters mu_mle and sigma_mle.
    
#     plt.plot(x, p, 'k', linewidth=2)
    
#     #plotting the fitted distribution
    
#     plt.title('Ca/Na Histogram and Fitted Log-Normal Distribution')
#     plt.xlabel('Ca/Na')
#     plt.ylabel('Density')
#     plt.savefig('chemweathering/figures/HistogramCaNa.png')
#     plt.close()
#     #plt.show()
    
    

    
    
    
    
    
    
    
#     return(processed_mean_mg_na, processed_sigma_mg_na, processed_mean_ca_na, processed_sigma_ca_na)



# def Xsil(df, processed_mean_mg_na, processed_sigma_mg_na, processed_mean_ca_na, processed_sigma_ca_na):
    
#     df_copy = df.copy()
    
#     df_copy = df_copy[~df_copy['unique_code'].str.contains('T1DW-0322', na=False)]
#     df_copy = df_copy[~df_copy['unique_code'].str.contains('RC15WF-1121', na=False)]
    
#     #print(df_copy)
    
#     #CaNa_sil = processed_mean_ca_na
    
#     CaNa_sil = 0.5
    
#     Ca_Na_sil_std = processed_sigma_ca_na
    
#     #MgNa_sil = processed_mean_mg_na
    
#     MgNa_sil = 0.5
    
#     Mg_Na_sil_std = processed_sigma_mg_na
#     #Galy and France Lanord say it is 0.2 on avg, we have gotten it primarily from bulk rock data, with a bit of log normal tweaking
    
    
#     df_copy['Ca_Sil'] = df_copy['*Na [aq] (mM)'] * CaNa_sil
    
#     df_copy['Mg_Sil'] = df_copy['*Na [aq] (mM)'] * MgNa_sil
    
#     df_copy['X_Sil'] = ((2*df_copy['Ca_Sil']) + (2*df_copy['Mg_Sil']) + df_copy['*K [aq] (mM)'] + df_copy['*Na [aq] (mM)'])/((2*df_copy['*Ca [aq] (mM)']) + (2*df_copy['*Mg [aq] (mM)']) + df_copy['*K [aq] (mM)'] + df_copy['*Na [aq] (mM)'])
    
#     #df_copy['X_Sil'] = ((df_copy['Ca_Sil']/2) + (df_copy['Mg_Sil']/2) + df_copy['*K [aq] (mM)'] + df_copy['*Na [aq] (mM)'])/((2*df_copy['*Ca [aq] (mM)']) + (2*df_copy['*Mg [aq] (mM)']) + df_copy['*K [aq] (mM)'] + df_copy['*Na [aq] (mM)'])
    
    
#     plt.figure(figsize=(10,6))
#     scatter = plt.scatter(df_copy['X_Sil'], df_copy['altitude_in_m'], alpha=0.7, s=70)
#     plt.xlabel('XSil')
#     plt.ylabel('Altitude')
    
#     plt.title(f'Scatter plot of Altitude vs. XSil - IF USE 0.5')

#             # Annotate each point with unique_code
#     for index, row in df_copy.iterrows():
#         plt.text(row['X_Sil'], row['altitude_in_m'], row['unique_code'], fontsize=8, ha='center', va='bottom')

#     #plt.legend()  # Include legend with labels
#     plt.savefig('chemweathering/figures/XSil.png') 
#     plt.show()  # Show each plot individually
#     plt.close()
#     return(df_copy)



# def Xsil_NEW(df, mg_na_sil_ratio, ca_na_sil_ratio, num_simulations=10000, cl_evap_min=0.01, cl_evap_max=10.0):
#     df_copy = df.copy()
    
#     # Filter out specific unique_codes
#     df_copy = df_copy[~df_copy['unique_code'].str.contains('T1DW-0322', na=False)]
#     df_copy = df_copy[~df_copy['unique_code'].str.contains('RC15WF-1121', na=False)]
    
#     # Initialize arrays to store results
#     optimal_cl_evap_values = []
#     optimal_X_Sil_values = []
    
#     # Define numerator factor
#     numerator_factor = 1 + 2 * mg_na_sil_ratio + 2 * ca_na_sil_ratio
    
#     for index, row in df_copy.iterrows():
#         best_cl_evap = None
#         best_X_Sil = -np.inf
        
#         for _ in range(num_simulations):
#             cl_evap = np.random.uniform(cl_evap_min, cl_evap_max)
#             na_sil = row['*Na [aq] (mM)'] - cl_evap
#             if na_sil < 0:
#                 continue
#             numerator = (na_sil * numerator_factor) + row['*K [aq] (mM)']
#             denominator = row['*Na [aq] (mM)'] + row['*K [aq] (mM)'] + 2 * row['*Mg [aq] (mM)'] + 2 * row['*Ca [aq] (mM)']
#             x_sil = numerator / denominator
            
#             if x_sil <= 1 and x_sil > best_X_Sil:
#                 best_X_Sil = x_sil
#                 best_cl_evap = cl_evap
        
#         optimal_cl_evap_values.append(best_cl_evap)
#         optimal_X_Sil_values.append(best_X_Sil)
    
#     # Store optimal values in the DataFrame
#     df_copy['optimal_cl_evap'] = optimal_cl_evap_values
#     df_copy['X_Sil_NEW'] = optimal_X_Sil_values
    
#     ## Plot the optimal values of cl_evap and X_Sil:
    
#     df_copy.dropna(subset=['optimal_cl_evap', 'X_Sil_NEW'], inplace=True)
    
#     # Plotting
#     plt.figure(figsize=(10, 6))
#     scatter = plt.scatter(df_copy['optimal_cl_evap'], df_copy['X_Sil_NEW'], alpha=0.7, s=70)
#     plt.xlabel('cl_evap')
#     plt.ylabel('X_Sil_NEW')
#     plt.title('Scatter plot of cl_evap vs. X_Sil_NEW')
    

#     # Annotate each point with unique_code, skipping rows where unique_code is 'nan'
#     for index, row in df_copy.iterrows():
#         plt.text(row['optimal_cl_evap'], row['X_Sil_NEW'], row['unique_code'], fontsize=8, ha='center', va='bottom')

#     plt.savefig('chemweathering/figures/cl_evap_vs_XSil.png')
#     plt.show()
#     plt.close()
    
#     return df_copy



# def Bathymetry_Old(df):

#     # Copy the DataFrame to avoid modifying the original one
#     df_copy = df.copy()
        
#     # Define the path to the DEM (Digital Elevation Model) raster file
#     dem_path = 'chemweathering/data/Raster.tif'
#     output = os.getcwd() + '/' + dem_path  # Adjusted path output
    
#     # Define bounds for clipping (actual coordinates)
#     west, south, east, north = -76.0, -14.0, -77.0, -13.0  # Example bounds (replace with actual coordinates)
    
#     west, south, east, north = bounds = west - .05, south - .05, east + .05, north + .05
    
#     elevation.clip(bounds=bounds, output=output, product='SRTM3')

        
#         # Execute elevation clipping (replace with actual function or process)
#         # Replace this line with your actual method or library to clip
#         # the elevation data to the specified bounds and save it to the output path.
#         # If you are using a specific library or tool for this, replace accordingly.
#         # For now, we'll set up the output path without the actual clipping.
#         # elevation.clip(bounds=bounds, output=output, product='SRTM3')
        
#     # Open the clipped DEM raster file (for illustration purposes)
#     dem_raster = rasterio.open('.' + dem_path)
        
#     src_crs = dem_raster.crs
#     src_shape = src_height, src_width = dem_raster.shape
#     src_transform = from_bounds(west, south, east, north, src_width, src_height)
#     source = dem_raster.read(1)    
        

#     # Define the destination CRS and transformation
#     dst_crs = {'init': 'EPSG:32718'}
#     dst_transform = from_origin(268000.0, 5207000.0, 250, 250)  # Adjusted coordinates
#     dem_array = np.zeros((451, 623))
#     dem_array[:] = np.nan
        
#     # Initialize an empty array to store reprojected DEM data
#     dem_array = np.zeros((src_height, src_width))
#     dem_array[:] = np.nan
        
#         # Reproject the source data to the destination CRS
#     reproject(
#         source=dem_raster.read(1),
#         destination=dem_array,
#         src_transform=dem_raster.transform,
#         src_crs=src_crs,
#         dst_transform=dst_transform,
#         dst_crs=dst_crs,
#         resampling=Resampling.bilinear
#         )
        
#     # Load colormap for visualization
#     topocmap = 'Spectral_r'
        
#         # Define vmin and vmax for color mapping
#     vmin = 180
#     vmax = 575
    
#     fig = plt.figure() 
#     ax = fig.add_subplot(1, 1, 1) 
        
#         # Plot the distribution of elevation data
#     ax = sns.histplot(dem_array.ravel(), axlabel='Elevation (m)')
#     ax = plt.gca()
        
#         # Apply colormap and adjust alpha for patches
#     #_ = [patch.set_color(topocmap(plt.Normalize(vmin=vmin, vmax=vmax)(patch.xy[0]))) for patch in ax.patches]
#     #_ = [patch.set_alpha(1) for patch in ax.patches]
        
#     # Save the figure
#     ax.get_figure().savefig('chemweathering/data/hello.png')



# def rotate_points_180(df_lons, df_lats, center_lon, center_lat):
#     # Rotate points by 180 degrees around the center
#     rotated_lons = 2 * center_lon - df_lons
#     rotated_lats = 2 * center_lat - df_lats
#     return rotated_lons, rotated_lats



# def Bathymetry(df):
#     # Copy the DataFrame to avoid modifying the original one
#     df_copy = df.copy()
    
#     # Define the path to the DEM (Digital Elevation Model) raster file
#     dem_path = 'chemweathering/data/output_SRTMGL1.tif'
    

    
#     output_path = os.path.join(os.getcwd(), dem_path)  # Adjusted path output

#     # Define bounds
#     west, south, east, north = -75.32426908093115, -13.526283371232736, -76.57091918634246, -11.749558766395864
#     bounds = (west - 0.05, south - 0.05, east + 0.05, north + 0.05)

#     # Calculate the center of the bounds
#     center_lon = (west + east) / 2
#     center_lat = (south + north) / 2

#     # Open the DEM raster file
#     with rasterio.open(output_path) as dem_raster:
#         src_crs = dem_raster.crs
        
#         # Clip the DEM to the specified bounds
#         out_image, out_transform = mask(dem_raster, [shapely.geometry.box(*bounds)], crop=True)
#         clipped_dem_array = out_image[0]

#         # Calculate the destination transform based on the bounds and the shape of the clipped array
#         dst_transform = from_bounds(*bounds, clipped_dem_array.shape[1], clipped_dem_array.shape[0])
        
#         # Define the destination CRS
#         dst_crs = 'EPSG:32718'

#         # Get the dimensions of the clipped DEM
#         height, width = clipped_dem_array.shape

#         # Create the destination transform and array
#         dst_transform, width, height = calculate_default_transform(
#             src_crs, dst_crs, width, height, *bounds
#         )
#         dst_array = np.zeros((height, width), dtype=np.float32) #define x and y and z

#         # Reproject the source data to the destination CRS
#         reproject(
#             source=clipped_dem_array,
#             destination=dst_array,
#             src_transform=out_transform,
#             src_crs=src_crs,
#             dst_transform=dst_transform,
#             dst_crs=dst_crs,
#             resampling=Resampling.bilinear
#         )

#     # Filter out elevation values below 0
#     dst_array[dst_array < 0] = np.nan # nice shorthand
#     print(f"Reprojected DEM shape: {dst_array.shape}")

#     # Generate the x and y coordinates in the projected CRS
#     x = np.linspace(bounds[0], bounds[2], dst_array.shape[1])
#     y = np.linspace(bounds[1], bounds[3], dst_array.shape[0])
#     x, y = np.meshgrid(x, y)
#     z = dst_array

#     # Interpolator for DEM elevations
#     interpolator = RegularGridInterpolator((y[:, 0], x[0, :]), z, bounds_error=False, fill_value=np.nan)
    
    

#     # Ensure the DataFrame coordinates are in the same CRS as the DEM
#     df_lons = df_copy['longitude_converted'].values
#     df_lats = df_copy['latitude_converted'].values

#     # Rotate the points by 180 degrees around the center
#     rotated_lons, rotated_lats = rotate_points_180(df_lons, df_lats, center_lon, center_lat)

#     # Adjust altitude values to be on or slightly above the DEM surface
#     #points = np.array([rotated_lats, rotated_lons]).T #for when plotting in 3D
    
#     points = np.array([df_lats, df_lons]).T
    
#     dem_alts = interpolator(points)  # Get DEM altitudes at the points
#     df_alts = dem_alts + 2000  # Adding 500 meters to ensure points are above the surface
    
#     # # Load the shapefile FOR 3D!
#     # shapefile_path = "/Users/enrico/Desktop/ROKOS Internship/QGIS/Map-Overlay.shp"
#     # gdf = gpd.read_file(shapefile_path)
    

#     # fig = plt.figure(figsize=(10, 6))
#     # ax = fig.add_subplot(111, projection='3d')

#     # # Plot the DEM surface
#     # surf = ax.plot_surface(x, y, z, cmap='Greys', edgecolor='none', alpha=0.5)
    
#     # # Plot the scatter points
#     # sc = ax.scatter(rotated_lons, rotated_lats, df_alts, alpha=0.9, s=100, c=df_copy['Na+K Normal (mM)'], cmap='coolwarm', label='Na+K Normal (mM)')

#     # # Add color bars
#     # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='DEM Elevation (m)')
#     # fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5, label='Na+K Normal (mM)')

#     # # Flatten the z-axis scale
#     # z_min, z_max = np.nanmin(z), np.nanmax(z)
#     # ax.set_zlim(z_min, z_max * 2)  # Adjust the divisor to control flattening

#     # Flip the DEM data and rotate it 180 degrees
#     z_flipped_rotated = np.flipud(np.fliplr(z))

#     # Load the shapefile
#     shapefile_path = "/Users/enrico/Downloads/Clipped Map Shapefile/Clipped_Map_Shapefile_New.shp"
#     gdf = gpd.read_file(shapefile_path)
    
#     # # Load the raster file for overlay
#     # overlay_path = "/Users/enrico/Desktop/ROKOS Internship/QGIS/GeologyTiff/Lithology3.tif"
#     # overlay_raster = rasterio.open(overlay_path)

#     # # Read the overlay data
#     # overlay_data = overlay_raster.read(1)

    
    
#     # Load second shapefile from Canete
#     shapefile_path2 = "/Users/enrico/Downloads/Canete shapefiles/Canete.shp"
#     gdf2 = gpd.read_file(shapefile_path2)
    
    
#     #For Shapefile
#     # Define color mapping based on ID
#     id_to_color = {
#         1: 'cyan',
#         4: 'cyan',
#         5: 'cyan',
#         7: 'cyan',
#         8: 'cyan',
#         2: 'red',
#         6: 'red',
#         3: 'fuchsia',
#         9: 'fuchsia'
#     }

#     # Define label mapping based on ID
#     id_to_label = {
#         1: 'sed',
#         4: 'sed',
#         5: 'sed',
#         7: 'sed',
#         8: 'sed',
#         2: 'volcanic',
#         6: 'volcanic',
#         3: 'plutonic',
#         9: 'plutonic'
#     }
    
#     # Define priority order (higher values will be plotted last and thus on top)
#     priority_order = {6: 3, 7: 2, 1: 1, 4: 1, 5: 1, 8: 1, 2: 1, 3: 1, 9: 1} #just because some of them do not show

#     # Sort the GeoDataFrame based on priority
#     gdf['priority'] = gdf['id'].map(priority_order)
#     gdf = gdf.sort_values(by='priority', ascending=True)


#     # Create the figure and axis
#     fig, ax = plt.subplots(figsize=(10, 6))

#     # Plot the DEM data in grayscale
#     c = ax.contourf(x, y, z_flipped_rotated, cmap='Greys', alpha=0.7)

#     # Plot the overlay shapefile without colours - GENERAL CODE FOR PLOTTING A SHAPEFILE
#     #gdf.plot(ax=ax, facecolor='none', edgecolor='blue', alpha=0.7, linewidth=1)
    
#     #### OLD SHAPEFILE PLOTTING CODE
#     # Plot the overlay shapefile with specified colors and labels
#     plotted_labels = set()  # Keeps track of which labels have been plotted
#     for geom, id_value in zip(gdf.geometry, gdf['id']):
#         color = id_to_color.get(id_value, 'grey')  # Default to grey if ID is not in the mapping
#         label = id_to_label.get(id_value, '')
#         if isinstance(geom, Polygon): ## This change ensures that geom is explicitly checked to be a Polygon type using the isinstance function, which is more Pythonic and handles the type check more explicitly.
#             x_poly, y_poly = geom.exterior.xy  # Extract the exterior coordinates (x and y) of the polygon
#             if label not in plotted_labels:
#                 ax.fill(x_poly, y_poly, color=color, alpha=0.5, edgecolor='black', linewidth=0.5, label=label)  # Fill the polygon with the specified color and add a label
#                 plotted_labels.add(label)
#             else:
#                 ax.fill(x_poly, y_poly, color=color, alpha=0.5, edgecolor='black', linewidth=0.5)  # Fill the polygon without adding a label if it has already been plotted
#         elif isinstance(geom, MultiPolygon):
#             for part in geom.geoms: #  In the original code, geom is treated as an iterable directly, which caused the TypeError. The revised code accesses the .geoms attribute of a MultiPolygon, which is an iterable of its constituent polygons, ensuring proper iteration.
#                 x_poly, y_poly = part.exterior.xy
#                 if label not in plotted_labels:
#                     ax.fill(x_poly, y_poly, color=color, alpha=0.5, edgecolor='black', linewidth=0.5, label=label)
#                     plotted_labels.add(label)
#                 else:
#                     ax.fill(x_poly, y_poly, color=color, alpha=0.5, edgecolor='black', linewidth=0.5)




#     # Plot second shapefile:
#     gdf2.plot(ax=ax, facecolor='none', edgecolor='yellow', alpha=1, linewidth=1, label='Canete Watershed')

#     # Plot the scatter points with different markers for each water body type
#     markers = {'mainstream': 'o', 'tributary': '^', 'spring': '*'}

#     # Normalize the optimal_cl_evap values
#     norm = mcolors.Normalize(vmin=df_copy['optimal_cl_evap'].min(), vmax=df_copy['optimal_cl_evap'].max())
#     cmap = plt.get_cmap('cividis')  # You can choose any colormap that transitions from dark to light

#     # Map the normalized values to colors
#     df_copy['edge_color'] = df_copy['optimal_cl_evap'].apply(lambda x: mcolors.to_hex(cmap(norm(x))) if not pd.isna(x) else 'black')


#     # Set vmin and vmax for the scatter plot color scale
#     vmin1, vmax1 = df_copy['X_Sil_NEW'].min(), df_copy['X_Sil_NEW'].max()
#     for water_body, marker in markers.items():
#         subset = df_copy[df_copy['water_body'] == water_body]
#         sc = ax.scatter(
#             subset['longitude_converted'].values,
#             subset['latitude_converted'].values,
#             c=subset['X_Sil_NEW'],
#             cmap='coolwarm',
#             s=100,
#             alpha=0.9,
#             edgecolor=subset['edge_color'].values,
#             marker=marker,
#             label=water_body,
#             vmax=vmax1,
#             vmin=vmin1,
#             linewidth=2
#         )
        
#     # Create legend with custom markers and labels
#     for water_body, marker in markers.items():
#         ax.scatter([], [], marker=marker, color='black', label=water_body)    
        
#          # Add color bar for edge color
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
#     cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=5, label='optimal_cl_evap')
       

#     # # Plot the geology overlay with original colors - FOR GEOLOGICAL MAP
#     # extent = [overlay_raster.bounds.left, overlay_raster.bounds.right, overlay_raster.bounds.bottom, overlay_raster.bounds.top]
#     # show(overlay_raster, ax=ax, extent=extent, alpha=1)

#     # # Close the overlay raster file
#     # overlay_raster.close()

#     # Add color bars
#     cbar = fig.colorbar(c, ax=ax, shrink=0.5, aspect=5, label='DEM Elevation (m)')
#     cbar_sc = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5, label='X_Sil - Gio')
    

#     # Set axis limits to match the DEM extent
#     ax.set_xlim(np.min(x), np.max(x))
#     ax.set_ylim(np.min(y), np.max(y))

#     # Labels
#     ax.set_xlabel('Longitude')
#     ax.set_ylabel('Latitude')

#     # Add legend
#     handles, labels = ax.get_legend_handles_labels()
#     unique_handles_labels = dict(zip(labels, handles))

#     # Add a custom entry for the Canete Watershed
#     custom_handle = plt.Line2D([0], [0], color='yellow', lw=2, label='Canete Watershed')
#     unique_handles_labels['Canete Watershed'] = custom_handle

#     ax.legend(loc='upper left', handles=unique_handles_labels.values())
    
#     ## add a title for this plot
#     ax.set_title('X_Sil - Gio -  Scatter Plot with DEM Overlay')


#     # Save the figure
#     plt.savefig('chemweathering/data/3d_surface_plot_with_overlay_gio_XSil.png')
#     plt.show()
#     plt.close(fig)

#     #OLDER CODE
#     # # Create a 3D surface plot using Plotly
#     # surface = go.Surface(x=x, y=y, z=z, colorscale='earth', colorbar=dict(title='Elevation (m)'))

#     # # Add scatter points from the DataFrame
#     # scatter = go.Scatter3d(
#     #     x=df_lons,
#     #     y=df_lats,
#     #     z=df_alts,
#     #     mode='markers',
#     #     marker=dict(
#     #         size=5,
#     #         color=df_copy['X_Sil'],
#     #         colorscale='Viridis',
#     #         opacity=0.8,
#     #         colorbar=dict(title='X_Sil')
#     #     )
#     # )

#     # # Create the layout
#     # layout = go.Layout(
#     #     title='3D Bathymetry with Scatter Points',
#     #     scene=dict(
#     #         xaxis_title='Longitude',
#     #         yaxis_title='Latitude',
#     #         zaxis_title='Elevation (m)',
#     #         zaxis=dict(range=[np.nanmin(z), np.nanmax(z)])  # Adjust the range if necessary
#     #     )
#     # )

#     # # Create the figure
#     # fig = go.Figure(data=[surface, scatter], layout=layout)

#     # # Save the plot to an HTML file
#     # output_html_path = 'chemweathering/data/3d_surface_plot_with_points.html'
#     # pio.write_html(fig, file=output_html_path, auto_open=False)

#     # # Show the plot
#     # fig.show()




# def NICB():
    
#     """The main NICB function to calculate NICB values for a given dataset"""
#     # Import data
#     df = pd.read_excel('chemweathering/data/Canete_Long_Data_Revisited_Local.xlsx', sheet_name='Data')
    
#     # Slice waters
#     df = df[df['sample_type'] == 'water']

#     # Slice relevant data
#     df_slice = slice_relevant_data(df)



#     # Replace non-numeric values with NaN in the dataframe except the first column and the last columns, which are unique_code and water_body
#     df_slice.loc[:, df_slice.columns[1:-1]] = df_slice.loc[:, df_slice.columns[1:-1]].apply(pd.to_numeric, errors='coerce')


#     # Create molar columns
#     df_neat = calculate_molar_concentration(df_slice)
    
#     #print(df_neat)

#     # Charge balance
#     df_neat = charge_balance(df_neat)

#     #print(df_neat)
    
#     # Initialize lists to store unique codes and NICB values 
#     NICB_Balance = NICB_Valid(df_neat)   
    
#     # Generate Gaillardet plot
#     GDPlot(df_neat, NICB_Balance)
    
#     #GD Plot converts the latitudes as well as the longitudes so must be included... to make more streamlined in future
    
#     map_with_log_colorscale(df_neat)
    
#     #Map also converts the latitudes as well as the longitudes so must be included
#     #Ideally would make this more streamlined. Why does it edit the df directly



#     # plot_NICB(df_neat)


    
#     #df_neat.to_csv('chemweathering/data/CorrectedValuesMatlab.csv', index=False)
    
#     df_piper = Piper(df_neat)
    
#     #Bathymetry(df_XSil)
    
#     df_corrected = Chloride_Correction2(df_neat, NICB_Balance) ## My correction
    
#     #df_corrected = Chloride_Correction(df_neat, NICB_Balance)
    
#     ###To uncomment for corrected value currections
    
#     mixdiag(df_corrected)
    
#     DistPlot(df_corrected)
    
#     processed_mean_mg_na, processed_sigma_mg_na, processed_mean_ca_na, processed_sigma_ca_na = RockData()
    
#     #df_XSil = Xsil_NEW(df_corrected, processed_mean_mg_na, processed_sigma_mg_na, processed_mean_ca_na, processed_sigma_ca_na)
    
#     df_XSil_NEW = Xsil_NEW(df_corrected, mg_na_sil_ratio=0.5, ca_na_sil_ratio=0.5)
    
#     Bathymetry(df_XSil_NEW)
    

    
#     #print(df_rock)



# if __name__ == '__main__':
#     NICB()





    



# ### Next step, use the new df-neat to plot some Gaillardet style plots

