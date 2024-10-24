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
from math import radians, cos, sin, sqrt, atan2




######### OUR CORRECTION CALCULATION #########



def NICB_Valid(df_neat):
    unique_codes_valid = []
    unique_codes_invalid = []
    nicb_valid = []
    nicb_invalid = []

    #Iterate through NICB values
    for index, i in df_neat['NICB'].items():
        unique_code = df_neat.loc[index, 'unique_code']
        if i > 0.1 or i < -0.1:
            nicb_invalid.append(i)
            unique_codes_invalid.append(unique_code)
        else:
            nicb_valid.append(i)
            unique_codes_valid.append(unique_code)

    # Calculate the maximum length to pad with None
    max_length = max(len(nicb_valid), len(nicb_invalid))

    # Pad lists with None to match the maximum length
    nicb_valid += [None] * (max_length - len(nicb_valid))
    nicb_invalid += [None] * (max_length - len(nicb_invalid))
    unique_codes_valid += [None] * (max_length - len(unique_codes_valid))
    unique_codes_invalid += [None] * (max_length - len(unique_codes_invalid))

    # Create the new DataFrame
    NICB_Balance = pd.DataFrame({
        'unique_code_valid': unique_codes_valid,
        'NICB_Valid': nicb_valid,
        'unique_code_invalid': unique_codes_invalid,
        'NICB_Invalid': nicb_invalid
    })  
    
    #print(len(NICB_Balance['unique_code_valid']))
            
    NICB_Balance.to_csv('chemweathering/data/ValidNICBRain.csv', index=False)

    return (NICB_Balance)



######### CALCULATING MOLAR CONCENTRATIONS AND CHARGE BALANCE FOR THE RAIN DATA #########

def molar_conc_seawater(df):
    """Calculate molar concentration of the major elements"""
    element_dict = {
        'molecules': ['Ca', 'Mg', 'K', 'Na', 'HCO3', 'Cl', 'SO4', 'Al', 'As', 'Ba', 'Bi', 'Ce', 'Fe', 'Li', 'Rb', 'Sr', 'CO3', 'H2PO4'],
        'molar_mass': [40.08, 24.31, 39.10, 22.99, 61.02, 35.45, 96.06, 26.98, 74.92, 137.33, 208.98, 140.12, 55.85, 6.94, 85.47, 87.62, 60.01, 97.99],
        'valency': [2, 2, 1, 1, 1, 1, 2, 3, 3, 2, 3, 3, 2, 1, 1, 2, 2, 1]
    }

    # Suffixes
    suffix = ' (ppm)'
    alt_suffix = ''

    df_copy = df.copy()

    for element, molar_mass, valency in zip(element_dict['molecules'],
                                            element_dict['molar_mass'],
                                            element_dict['valency']):
        # Check if the standard or alternate column exists
        if element + suffix in df_copy.columns:
            col_name = element + suffix
        elif element + alt_suffix in df_copy.columns:
            col_name = element + alt_suffix
        else:
            # Skip element if neither column exists
            continue

        # Create new molar columns
        df_copy.loc[:, element + ' [aq] (mM)'] = df_copy[col_name] / molar_mass
        df_copy.loc[:, element + ' [aq] (meq/L)'] = df_copy.loc[:, element + ' [aq] (mM)'] * valency

    #print(df_copy.columns)

    # Select relevant columns
    #df_copy = [ [col for col in df_copy.columns if ' [aq] (mM)' in col] + [col for col in df_copy.columns if ' [aq] (meq/L)' in col] ]

    #print(df_copy)

    return df_copy

def charge_balance_modded(df):
    """Function to charge balance"""
    cations = ['Ca', 'Mg', 'K', 'Na', 'Al', 'Ba', 'Fe', 'Li']
    anions = ['NO3', 'Cl', 'SO4']
    
    # Debug: Check which columns are being used for summation
    cation_columns = [col for col in df.columns if any(cation in col for cation in cations) and ' [aq] (meq/L)' in col]
    anion_columns = [col for col in df.columns if any(anion in col for anion in anions) and ' [aq] (meq/L)' in col]

    
    #print(f"Cation columns: {cation_columns}")
    #print(f"Anion columns: {anion_columns}")


    # Calculate the sum of cations and anions
    # Calculate the sum of cations and anions
    df['sum_cations'] = df[cation_columns].sum(axis=1)
    df['sum_anions'] = df[anion_columns].sum(axis=1)
    ####### Instead of adding 0.22, we will calculate the NICB difference and sum #######
    ####### This will come out to give an NICB of more than 0.1 in most cases. We will then create a new column rain_processed['HCO3 [aq] (mM)'] to make sure that the NICB is equal to zero #######


    # NICB calculation
    df['NICB diff'] = (df['sum_cations'] - df['sum_anions'])
    df['NICB sum'] = (df['sum_cations'] + df['sum_anions'])

    # Remove rows with NICB sum = 0
    #df = df[df['NICB sum'] != 0]

    df['NICB'] = df['NICB diff'] / df['NICB sum']

    return df


def charge_balance_modded_HCO3(df):
    """Function to charge balance"""
    cations = ['Ca', 'Mg', 'K', 'Na', 'Al', 'Ba', 'Fe', 'Li']
    anions = ['NO3', 'Cl', 'SO4', 'HCO3']
    
    
    # Calculate the sum of cations and anions
    df['sum_cations'] = df[[col for col in df.columns if any(cation in col for cation in cations) and ' [aq] (meq/L)' in col]].sum(axis=1)

    df['sum_anions'] = df[[col for col in df.columns if any(anion in col for anion in anions) and ' [aq] (meq/L)' in col]].sum(axis=1)

    ####### Instead of adding 0.22, we will calculate the NICB difference and sum #######
    ####### This will come out to give an NICB of more than 0.1 in most cases. We will then create a new column rain_processed['HCO3 [aq] (mM)'] to make sure that the NICB is equal to zero #######


    # NICB calculation
    df['NICB diff'] = (df['sum_cations'] - df['sum_anions'])
    df['NICB sum'] = (df['sum_cations'] + df['sum_anions'])

    # Remove rows with NICB sum = 0
    #df = df[df['NICB sum'] != 0]

    df['NICB'] = df['NICB diff'] / df['NICB sum']

    return df

######### REFER TO OTHER CHLORIDE CORRECTIONS FOR IN DEPTH EXPLANATION #########


def adjust_HCO3_to_balance(df):
    """Adjust HCO3 to balance NICB for each rain sample."""
    df['HCO3 [aq] (meq/L)'] = 0
    NICB_Balance = NICB_Valid(df)
    
    #print(NICB_Balance)

    # Directly access the single value in NICB_Invalid for the condition
    while (df['NICB'] > 0.05).any() or (df['NICB'] < -0.05).any():
        df['HCO3 [aq] (meq/L)'] += 0.01
        df = charge_balance_modded_HCO3(df)
        
        #print('processing')

    return df



def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c





######### OUR CORRECTION #########

def Chloride_Correction4(df, df2):


    ######### RAIN DATA APPENDING #########
    
    rain = pd.read_excel('chemweathering/agilentexporttools/Rain_Ions.xlsx', sheet_name='Conconcentration (Processed)')
        
    # Convert concentrations to molar units (assuming molar_conc_seawater function is defined)
    rain_processed = molar_conc_seawater(rain)
    
    ## Change the first column, first row name to 'unique_code'
    rain_processed.columns = ['unique_code'] + [col for col in rain_processed.columns[1:]]
    
    # Assign a numeric ID to each rain sample
    rain_processed['rain_sample_id'] = range(len(rain_processed))
    
    # Starts at zero
    
    rain_processed = charge_balance_modded(rain_processed)
    
    #remove last row of rain_processed:
    rain_processed = rain_processed.iloc[:-1]
    
    rain_dict = {}
    for i, sample in rain_processed.iterrows():
          sample_df = sample.to_frame().T
          #print(sample_df)
          sample_df = adjust_HCO3_to_balance(sample_df)
          rain_dict[sample['rain_sample_id']] = sample_df

    # Convert rain_dict into a dataframe:
    rain_df = pd.concat(rain_dict.values(), keys=rain_dict.keys()).reset_index(level=1, drop=True)
    
    rain_df['HCO3 [aq] (mM)'] = rain_df['HCO3 [aq] (meq/L)']
    
    

    
    
    
    ######### DATAFRAME MANIPULATION #########
    
    df_copy = df.copy()
    
    # Filter df_copy to include only valid charge balanced  samples
    df_copy = df_copy[df_copy['unique_code'].isin(df2['unique_code_valid'])]
    
    #############################################
    
    na_before = df_copy['Na [aq] (mM)']
    

    #cl_minimum = the lowest cl value in df_copy:
    cl_minimum = df_copy['Cl [aq] (mM)'].min()
    

    ########################### RAIN CORRECTION CALCULATION ###########################

    # Correction calculation
    
    
    # Potential to add Evapotranspiration if wanted
    
     #### CHANGE ####
    ##### Need to change this so that it does it for every rain sample, and then adds a starred column to df copy for every rain sample #####
    
    #print(df_copy.columns)
    
    
    # Iterate over each element (column) in the rain_processed DataFrame
    for element in rain_df.columns:
        
        if '[aq] (mM)' in element:
            
            if element in df_copy.columns:
                
                
                # Get the value of the current element from the rain_processed DataFrame (assuming there's only one value)
                
                for i, sample in rain_df.iterrows():
                    
                    rain_element_value = sample[element]
                    
                    # Get the value of 'Cl [aq] (mM)' from the rain_processed DataFrame (assuming there's only one value)
                    
                    cl_rain_value = sample['Cl [aq] (mM)']
                    
                    # Perform the correction calculation:
                    # For each row in df_copy, subtract the rain element value from the river element value
                    df_copy['*' + str(sample['rain_sample_id']) + element] = df_copy[element] - rain_element_value 
            else:
                # Print a message if the current element is not found in the df_copy DataFrame
                print(f"Element {element} not found in df_copy")

    
    #print(df_copy.columns)
    #
    
    #################################################################################
    
    # Identify row number for RC15WF-1121 in unique code:
    #row_number = df_copy[df_copy['unique_code'] == 'RC15WF-1121'].index[0]
    # row_number = df_copy[df_copy['unique_code'] == 'T1BWF-1121'].index[0]
    
    # if '*0Cl [aq] (mM)' in df_copy.columns:
    #     print('Cl for 0: ' + str(df_copy.loc[row_number, '*0Cl [aq] (mM)']))
    #     print('Cl for 2: ' + str(df_copy.loc[row_number, '*2Cl [aq] (mM)']))
    #     print('Cl for 3: ' + str(df_copy.loc[row_number, '*3Cl [aq] (mM)']))

        
    ######### EVAPORITE ENDMEMBER RATIO CALCULATION #########
    
    # Another time also change this so that it does it for every evaporite sample, and then adds a starred column to df copy for every rain sample
    # For samples with water body type 'spring', calculate all element/Cl ratios for the corrected "*" values and store them in a dictionary:
    spring_ratios = {}

    # Get the row for the spring sample
    #spring_row = df_copy[df_copy['unique_code'] == 'T1AWF-1121'] 
    
    ######### PICKING COWLICK DEATH ROCK #########
    
    spring_row = df_copy[df_copy['unique_code'] == 'T1DW-0322']
    
    
    ## Print the spring row:

    
    
    # Iterate over each element in rain_processed
    # Initialize an empty DataFrame to store the ratios
    ratio_columns = [col.replace(' [aq] (mM)', '') + '/Cl' for col in rain_df.columns if '[aq] (mM)' in col and col != 'Cl [aq] (mM)']
    spring_ratios_df = pd.DataFrame(index=range(0, 6), columns=ratio_columns)
    
    

    # Iterate over each element in rain_processed
    for element in rain_df.columns:
        
        if '[aq] (mM)' in element and element != 'Cl [aq] (mM)':
            
            # Strip out '[aq] (mM)' and prepare the column name for the ratio DataFrame
            ratio_column_name = element.replace(' [aq] (mM)', '') + '/Cl'
            
            for i in range(0, 6):
                
                if '*' + str(i) + element in df_copy.columns and '*' + str(i) + 'Cl [aq] (mM)' in df_copy.columns:
                    
                    # Calculate the ratio for the current element
                    element_value = spring_row['*' + str(i) + element].values[0]  # Convert to scalar - have to
                    cl_value = spring_row['*' + str(i) + 'Cl [aq] (mM)'].values[0]  # Convert to scalar
                    ratio = element_value / cl_value
                    
                    # Store the ratio in the DataFrame
                    spring_ratios_df.at[i, ratio_column_name] = ratio

    # Display the resulting DataFrame with spring ratios
    #print(spring_ratios_df.columns)

    
    # Remove columns that contain NaN values
    spring_ratios_df = spring_ratios_df.dropna(axis=1)
    
 
  
            
    for index, row in spring_ratios_df.iterrows():
        
        for col in spring_ratios_df.columns:  # iterate over each column

                spring_ratios_df.at[index, col] = row[col] / row['Na/Cl']
                
    
           
        
        
    #print(spring_ratios_df)   
    
    
    
    
    
    
    # Now have spring ratios for every rain.  
        
              


    # # Calculate the mean ratio for each element from the spring sample
    # mean_spring_ratios = {element: ratio for element, ratio in spring_ratios.items()}
    
    # # Iterate over each element in mean_spring_ratios
    # for element, ratio in mean_spring_ratios.items():
    #     # Divide the ratio by the initial Na/Cl ratio
    #     mean_spring_ratios[element] /= mean_spring_ratios['Na [aq] (mM)']
    
    #print(mean_spring_ratios)


    # ################################################################
    


    # ######### CORRECT USING THE SPRING RATIOS #########
        
    
    ## The spring ratios are now the ratios we will use to correct once more:
    
      #### CHANGE ####   
    #### Change this so that it iterates per rain sample and then adds a '+'column to df_copy for each rain sample
    
    # Apply the spring ratios to correct the elements for the remaining Cl amount

    # Output the corrected DataFrame
    #print(df_copy)
    
    # Iterate over each element in rain_processed
    
    # Print or return the modified DataFrame
    #print(df_copy['+0Na [aq] (mM)'])
    
    # ######################################################
    
    # Match the element in df_copy: For example, match Ca in Ca/Cl from spring_ratios_df with *iCa [aq] (mM) in df_copy where i varies from 0 to 7.

    # Extract the ratio from spring_ratios_df: The ratio corresponding to the element (like Ca/Cl).

    # Perform the calculation: Subtract the product of the ratio and the corresponding Cl [aq] (mM) value from *iCa [aq] (mM) and store it in a new column.
        

    
    import re

    #Create a dictionary to hold the new columns
    new_columns = {}

    # Iterate over each element in df_copy (e.g., *iCa [aq] (mM))
    for element in df_copy.columns:
        # Check if the column matches the pattern '*iElement [aq] (mM)'
        match = re.search(r'\*\d+([A-Za-z0-9]+) \[aq\] \(mM\)', element)
        
        # r before the quote indicates the string is a raw string. this allows python to treat backslashes as literal characters
        # \* tells python that we are looking for the star character in particular, as if the \ was not added star would signify a command in the search
        # \d matches any digit 0-9
        # the + following \d means one or more digits, so this will match any sequence of digits to length one or more
        # A-Za-z0-9 matches any uppercase or lowercase letter or number, and once again the + means use one or more of these characters
        # the parentheses around the A-Za-z0-9 make a capture group. This captures the element name e.g., Ca, Mg, HCO3, SO4
        # \[aq\] matches the literal string aq. Make sure to add the space before as that matches the actual space too
        
        if match:
            # Extract the element symbol (e.g., Ca, Mg, HCO3, SO4, etc.)
            element_symbol = match.group(1)
            
            # Construct the ratio column name in spring_ratios_df (e.g., Ca/Cl)
            ratio_col_name = element_symbol + '/Cl'
            
            if ratio_col_name in spring_ratios_df.columns:
                # Iterate over the index values (0 to 7)
                for i in range(0, 6):
                    col_name = f'*{i}{element_symbol} [aq] (mM)'
                    cl_col_name = f'*{i}Cl [aq] (mM)'
                    
                    if col_name in df_copy.columns and cl_col_name in df_copy.columns:
                        # Retrieve the ratio from spring_ratios_df (assuming the rows are aligned)
                        ratio = spring_ratios_df.loc[index, ratio_col_name]
                        
                        # Calculate the new value
                        new_col_name = f'+{i}{element_symbol} [aq] (mM)'
                        new_columns[new_col_name] = df_copy[col_name] - (ratio * df_copy[cl_col_name])

    # Once all calculations are done, concatenate the new columns to the original DataFrame
    df_copy = pd.concat([df_copy, pd.DataFrame(new_columns)], axis=1)
    
    
    
    #print(rain_df)

        
    ### NOW need to correct for the rain based on its location
    
    # Change accordingly. 0 and 3 correspond to Villa de arma 27 and Refugio V Rainwater
    # MHP 055 is 2
    rain_points = {
    "0": {"lat": -12.298422, "long": -75.814397, "index": 0},
    "2": {"lat": -12.170604, "long": -75.628256, "index": 2},
    "3": {"lat": -12.847908, "long": -75.583624, "index": 3}
    }
    
    #Villa De Arma
    #-12.298422, -75.814397
    
    
    # Calculate the distances to each rain sample
    for index, sample in df_copy.iterrows():
        distances = {}
        for rain_sample_id, rain_data in rain_points.items():
            distance = haversine(sample['latitude_converted'], sample['longitude_converted'], rain_data['lat'], rain_data['long'])
            distances[rain_sample_id] = distance
            
        # Store the distances in the DataFrame
        for rain_sample_id in rain_points.keys():
            df_copy.at[index, f'{rain_sample_id}_distance'] = distances[rain_sample_id]

    # Identify the closest rain sample for each row
    distance_columns = [f'{rain_sample_id}_distance' for rain_sample_id in rain_points.keys()]
    df_copy['closest_rain_sample_id'] = df_copy[distance_columns].idxmin(axis=1).str.extract(r'(\d+)')[0]
        
    

    ###### Now to filter out the non closest
    
    # Define the cations and anions
    
    # This is a work in progress
    
    # Remove all columns that have '[aq]' in them but do not have a '*' or '+' sign before '[aq]'
    
    #df_copy = df_copy.loc[:, df_copy.columns.str.contains(r'(\*\w+ \[aq\])|(\+\w+ \[aq\])') | ~df_copy.columns.str.contains(r'\[aq\]')]
    # don't want to use that because the str.contains method is interpreting the regular expression as containing the match group due to the parentheses ()
    # These match groups are not being used after so it gives me a warning. Instead use the following without the parentheses
    
    df_copy = df_copy.loc[:, df_copy.columns.str.contains(r'\*\w+ \[aq\]|\+\w+ \[aq\]') | ~df_copy.columns.str.contains(r'\[aq\]')]

    
    
    
    # r before the quote indicates the string is a raw string. this allows python to treat backslashes as literal characters
    # \* tells python that we are looking for the star character in particular, as if the \ was not added star would signify a command in the search
    # \+ also tells python we are looking at a literal plus character
    # \w+ means we are looking at one or more word characters
    # the pipe operator | means or
    
    # Remove all columns that have a star (*) in them
    df_copy = df_copy.loc[:, ~df_copy.columns.str.contains(r'\*')]

    # List of columns to exclude from removal
    columns_to_keep = ['calculated_discharge_in_m3_s-1', 'Alkalinity_CaCO3_in_mg_kg-1']

    # Additionally, include specific columns like 'SO4 [aq] (mM)' that should be kept
    # Adjust the following list as necessary based on the exact names of the columns in your DataFrame
    additional_columns_to_keep = df_copy.columns[df_copy.columns.str.contains(r'SO4')]

    # Combine the columns to keep
    columns_to_keep.extend(additional_columns_to_keep)

    # Create a boolean mask for columns that contain any of the digits 1, 2, 4, 5, 6, 7
    columns_with_digits = df_copy.columns.str.contains(r'[1456]')

    # Create a boolean mask for columns that should be kept (i.e., not removed)
    columns_to_keep_mask = df_copy.columns.isin(columns_to_keep)

    # Combine the masks: Keep columns that don't match the digits or are explicitly listed to keep
    df_copy = df_copy.loc[:, ~columns_with_digits | columns_to_keep_mask]
    # df.loc[rows, columns]

    #df_copy.to_excel('chemweathering/data/testing_correction_4.xlsx')
        
    
    
    # Row by row, compare the closest_rain_sample_id as an integer to the columns that have [aq] in them and a number in them.
    
    # Iterate over each row in df_copy
    for index, row in df_copy.iterrows():
        # Iterate over each column in df_copy
        for column in df_copy.columns:
            if '[aq]' in column and '+' in column:
                
                # Matching the pattern for the element name and the number after the '+'
                match2 = re.search(r'\+(\d+)([A-Za-z0-9]+) \[aq\] \(mM\)', column)
                
                if match2:
                    # Extract the number closest to the '+' and the element symbol
                    number = int(match2.group(1))
                    element_symbol = match2.group(2)
                    
                    # Compare the extracted number with closest_rain_sample_id in the current row
                    if number == int(row['closest_rain_sample_id']):
                        # Construct the new column name
                        new_column_name = f'+{element_symbol} [aq] (mM)'
                        
                        # Check if the new column already exists, if not create it with None values
                        if new_column_name not in df_copy.columns:
                            df_copy[new_column_name] = pd.Series([None] * len(df_copy))
                            
                        # Assign the value from the original column to the new column
                        df_copy.at[index, new_column_name] = row[column]
    
    
    #df_copy.to_excel('chemweathering/data/testing_correction_4_new.xlsx')    
    
    
    
    # List the columns that have [aq] in them AND a number after the '+'. Extract the number from these columns.
    # Initialize an empty list to collect columns to remove
    columns_to_remove = []

    # Iterate over each column in df_copy
    for column in df_copy.columns:
        if '[aq]' in column and '+' in column:
            # Extract the first digit that comes after the '+'
            match = re.search(r'\+(\d)', column)  # Only match the first digit after the '+'
            if match:
                number = int(match.group(1))
                # If you want to remove columns with any specific digit(s), add them to the removal list
                # In this example, we remove all columns with the first digit being '1', '2', '3', etc.
                if number in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:  # Modify this list to specify which digits to remove
                    columns_to_remove.append(column)

    # Drop the columns that match the criteria
    df_copy = df_copy.drop(columns=columns_to_remove)
    
    # df_mock returns the evaporite corrected
    df_mock = df_copy[['unique_code'] + [col for col in df_copy.columns if '+' in col] + ['latitude_converted', 'longitude_converted', 'NICB', 'calculated_discharge_in_m3_s-1', 'Alkalinity_CaCO3_in_mg_kg-1', 'field_pH', 'altitude_in_m', 'water_body']]

    
    
    
    
    
    # If negative value not in +Ba [aq] (mM)	+Fe [aq] (mM)	+Li [aq] (mM)	+Sr [aq] (mM), then remove that row
    
    # Then Convert negative values to NaNs
        
    # Define the columns where negative values should not cause the row to be removed
    columns_to_exclude = ['+Ba [aq] (mM)', '+Fe [aq] (mM)', '+Li [aq] (mM)', '+Sr [aq] (mM)']

    # Initialize a set to collect unique codes where rows are removed
    unique_codes = set()

    # Iterate over each column in df_mock
    for col in df_mock.columns:
        if '+' in col:
            if col not in columns_to_exclude:
                # Identify rows with negative values in columns that are not in the excluded list
                negative_samples = df_mock[df_mock[col] < 0]
                
                # Collect unique codes for rows to be removed
                unique_codes.update(negative_samples['unique_code'])
                
                # Remove rows with negative values in columns that are not excluded
                df_mock = df_mock[df_mock[col] >= 0]

    # Convert remaining negative values to NaNs in the excluded columns
    for col in columns_to_exclude:
        if col in df_mock.columns:
            df_mock[col] = df_mock[col].apply(lambda x: x if x >= 0 else pd.NA)
            
            
    columns_to_remove_outright = ['+Ba [aq] (mM)', '+Fe [aq] (mM)', '+Li [aq] (mM)']     
    
    df_mock = df_mock.drop(columns=columns_to_remove_outright) 

    # Output the unique codes where rows were removed
    print("Unique codes with negative values that were removed:")
    print(unique_codes)
    
    #df_mock.to_excel('chemweathering/data/testing_correction_4.xlsx')
    
    return df_mock
    
