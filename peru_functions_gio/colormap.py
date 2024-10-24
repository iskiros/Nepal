import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geodatasets
import geopandas as gpd
import re
import folium
from folium import plugins
import rioxarray as rxr
import earthpy as et
import earthpy.spatial as es
import webbrowser
from branca.colormap import LinearColormap  # Correct import for LinearColormap

# Load the simplified data
data = pd.read_excel('chemweathering/data/Canete_Long_Data_Revisited_Local.xlsx', sheet_name='Data')

datacopy = data

# Filtering the data
pattern = '-|_'
filtered_dc = datacopy[datacopy['unique_code'].str.contains(pattern, na=False)]

ColumnIDO = filtered_dc['unique_code']
array = [i.partition("-")[0] for i in ColumnIDO]
filtered_dc['Location2'] = array

# Convert dates
filtered_dc['date'] = pd.to_datetime(filtered_dc['date'])
filtered_dc['formatted_date'] = filtered_dc['date'].dt.strftime('%d-%m%y')
filtered_dc['SampleID'] = filtered_dc['Location2'] + '-' + filtered_dc['formatted_date']


#print(filtered_dc['SampleID'])

# Function to convert degrees and minutes to decimal degrees for latitude
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

# Function to convert degrees and minutes to decimal degrees for longitude
def convert_longitude(row):
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

# Apply conversions
# Apply conversions







filtered_dc['latitude_converted'] = filtered_dc.apply(convert_latitude, axis=1)
filtered_dc['longitude_converted'] = filtered_dc.apply(convert_longitude, axis=1)

# Drop rows with NaN values in converted latitude and longitude
filtered_dc.dropna(subset=['longitude_converted', 'latitude_converted'], inplace=True)


# # Filter to include only water samples
# filtered_dc = filtered_dc[filtered_dc['sample_type'].str.contains('water', na=False)]

# Filter to exclude samples containing 'sediment' or 'rock' in sample_type
filtered_dc = filtered_dc[~filtered_dc['sample_type'].str.contains('sediment|rock', na=False)]




# Ensure 'Na_in_mg_kg-1' is numeric and drop rows with NaN in 'Na_in_mg_kg-1'
filtered_dc['Na_in_mg_kg-1'] = pd.to_numeric(filtered_dc['Na_in_mg_kg-1'], errors="coerce")
filtered_dc.dropna(subset=['Na_in_mg_kg-1'], inplace=True)


#print((filtered_dc['SampleID'])) Checking it has worked correctly, and that I can see the newest data



data_locations = filtered_dc[["latitude_converted", "longitude_converted", "SampleID", "Na_in_mg_kg-1"]]

#########
#########




#########
#########
#########
# Create a map using Folium
m = folium.Map(location=[data_locations.latitude_converted.mean(), data_locations.longitude_converted.mean()], zoom_start=10)

tile = folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri',
    name='Esri Satellite',
    overlay=False,
    control=True
).add_to(m)

# Define the colormap correctly
colormap = LinearColormap(colors=['blue', 'green', 'yellow'], vmin=0, vmax=30)  # Adjust vmax to match the data range

colormap.add_to(m)

legend_html = '''
     <div style="position: fixed; 
                 top: 10px; right: 25px; width: 500px; height: 40px; 
                 border:2px solid grey; z-index:9999; font-size:14px;
                 background-color: white; opacity: 0.5;
                 ">
     &nbsp; <strong>Na (ppm)</strong> <br>
     &nbsp; 0 - 30 <br>
        </div>
     '''

m.get_root().html.add_child(folium.Element(legend_html))

for index, location_info in data_locations.iterrows():
    value = location_info['Na_in_mg_kg-1']
    # Ensure the value is within the range of the colormap
    if 0 <= value <= 30:
        folium.CircleMarker(
            [location_info["latitude_converted"], location_info["longitude_converted"]],
            popup=f"Na (ppm): {location_info['Na_in_mg_kg-1']}<br> <br> Sample: {location_info['SampleID']}",
            radius=10,
            color="black",
            weight=1,
            fill_opacity=0.6,
            opacity=1,
            fill_color=colormap(value),
            fill=True
        ).add_to(m)







#print((data_locations['SampleID']))#

# Save the map to the specified folder
folder_path = 'chemweathering/data/figures'
file_name = 'mapgio.html'
os.makedirs(folder_path, exist_ok=True)
full_path = os.path.join(folder_path, file_name)
m.save(full_path)
