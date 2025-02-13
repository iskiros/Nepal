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
from matplotlib import rcParams, font_manager
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

# your data
df = pd.read_csv('data/geochem_data.csv')



# Prepare the input string for PHREEQC
input_file_content = ''

# Dictionary to link solution number to Sample ID
solution_to_sample = {}

for idx, row in df.iterrows():
    
    # Conversion from µM to ppm for CaCO3
    alkalinity_ppm = row['Alkalinity'] * 100.09 / 1000  # Convert µM to ppm and correct for the molarity of the acid
    
    # Solution block
    solution_number = idx + 1  # This is the solution number (1-based)
    sample_id = row['Sample ID']  # Sample ID from the dataframe
    
    # Link solution number to Sample ID in the dictionary
    solution_to_sample[solution_number] = sample_id

    solution_block = f"""
TITLE {sample_id} -- Geochemical Solution
SOLUTION {solution_number} {sample_id}
    units ppm
    pH {row['pH']}
    temp {row['Temperature']}
    Ca {row['Ca_ppm']}
    Mg {row['Mg_ppm']}
    Na {row['Na_ppm']}
    Al {row['Al_ppm']}
    K {row['K_ppm']}
    Fe {row['Fe_ppm']}
    Mn {row['Mn_ppm']}
    Si {row['Si_ppm']}
    Alkalinity {row['Alkalinity']} as CaCO3
    S(6) {row['S_ppm']}
"""

# ignoring chloride... for now becase not all samples have chloride
    input_file_content += solution_block
    
# Add the EQUILIBRIUM_PHASES block for minerals of interest
input_file_content += """
EQUILIBRIUM_PHASES 1
    Calcite 0.0 0.0   # Saturation index for calcite
    Gypsum 0.0 0.0    # Saturation index for gypsum
    Quartz 0.0 0.0    # Saturation index for quartz
    Halite 0.0 0.0    # Saturation index for halite
    Dolomite 0.0 0.0  # Saturation index for dolomite
    Anhydrite 0.0 0.0 # Saturation index for anhydrite
    Pyrite 0.0 0.0    # Saturation index for pyrite
    Hematite 0.0 0.0  # Saturation index for hematite
    Siderite 0.0 0.0  # Saturation index for siderite
    Kaolinite 0.0 0.0 # Saturation index for kaolinite
    Gibbsite 0.0 0.0  # Saturation index for gibbsite
    Goethite 0.0 0.0  # Saturation index for goethite
    Anorthite 0.0 0.0 # Saturation index for anorthite
    Albite 0.0 0.0    # Saturation index for albite
    Illite 0.0 0.0    # Saturation index for illite

SELECTED_OUTPUT
    -file results.txt
    -saturation_indices
"""

# Write the input file to disk
with open("phreeqc_input_test.pqi", "w") as f:
    f.write(input_file_content)

print("Input file generated: phreeqc_input.pqi")

# Optionally print the solution-to-sample mapping for verification
print("Solution to Sample ID mapping:", solution_to_sample)


##########

# Extracting example

import re
import pandas as pd

# Read the file content
with open("phreeqc_Al_sens_test.txt", "r") as file:
    content = file.read()

# Split solutions
solution_numbers = re.findall(r"Initial solution (\d+)\.", content)
solution_blocks = re.split(r"Initial solution \d+\.", content)[1:]  # Skip preamble

data = []

for solution_number, block in zip(solution_numbers, solution_blocks):
    print(f"\nProcessing solution {solution_number}...")

    # Extract "Distribution of species"
    dist_species_match = re.search(r"Distribution of species(.+?)Saturation indices", block, re.DOTALL)
    dist_species = {}

    if dist_species_match:
        species_data = dist_species_match.group(1).strip()
        print(f"Species data found for solution {solution_number}:\n{species_data[:200]}")  # Debug
        
        # Parse each line in the species data
        for line in species_data.split("\n"):
            # Adjust the regex to capture the 5th column (log of activity)
            match = re.match(r"^\s*(\S+)\s+\S+\s+\S+\s+\S+\s+(-?\d+\.\d+)\s+.*$", line)
            if match:
                species, log_activity = match.groups()
                dist_species[f"log_activity_{species}"] = log_activity
        
        print(f"Extracted species activity: {dist_species}")
    else:
        print(f"No Distribution of species section found for solution {solution_number}.")

    # Extract "Saturation indices" data
    saturation_match = re.search(r"Saturation indices(.+?)Na2SO4", block, re.DOTALL)

    saturation_indices = {}
    if saturation_match:
        # Extract the data after the header row, which we will skip
        saturation_data = saturation_match.group(1).strip()
        print(f"Saturation data found for solution {solution_number}:\n{saturation_data[:200]}")  # Debug

        # Split the data into lines, skipping the header row
        lines = saturation_data.split("\n")[1:]  # Skip the first row (header)

        # Parse each row in the saturation indices section
        for line in lines:
            match = re.match(r"^\s*(\S+)\s+(-?\d+\.\d+)\s+.*$", line)  # Match phase and SI value
            if match:
                phase, si = match.groups()
                saturation_indices[f"SI_{phase}"] = si
        
        print(f"Extracted saturation indices for solution {solution_number}: {saturation_indices}")
    else:
        print(f"No valid Saturation indices section found for solution {solution_number}.")

    # Combine data
    combined_data = {"solution_number": solution_number}
    combined_data.update(dist_species)
    combined_data.update(saturation_indices)
    data.append(combined_data)
    
# Put data into DataFrame
# df_pqc = pd.DataFrame(data)
df_pqc_sens_test = pd.DataFrame(data)


# Display the DataFrame
print(df_pqc_sens_test)

df_pqc_sens_test.to_excel('hiphreeqc_output_altest.xlsx', index=False)
