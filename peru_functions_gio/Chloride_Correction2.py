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
from peru_functions_gio.NICB_Valid import NICB_Valid



######### OUR CORRECTION CALCULATION #########


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
    cations = ['Ca', 'Mg', 'K', 'Na', 'Al', 'As', 'Ba', 'Bi', 'Fe', 'Li', 'Rb', 'Sr']
    anions = ['HCO3', 'Cl', 'SO4', 'H2PO4', 'CO3']

    # Calculate the sum of cations and anions
    df['sum_cations'] = df[[col for col in df.columns if any(cation in col for cation in cations) and ' [aq] (meq/L)' in col]].sum(axis=1)

    df['sum_anions'] = df[[col for col in df.columns if any(anion in col for anion in anions) and ' [aq] (meq/L)' in col]].sum(axis=1) + 0.22

    ####### ADDING 0.22 here #######


    # NICB calculation
    df['NICB diff'] = (df['sum_cations'] - df['sum_anions'])
    df['NICB sum'] = (df['sum_cations'] + df['sum_anions'])

    # Remove rows with NICB sum = 0
    df = df[df['NICB sum'] != 0]

    df['NICB'] = df['NICB diff'] / df['NICB sum']

    return df

######### REFER TO OTHER CHLORIDE CORRECTIONS FOR IN DEPTH EXPLANATION #########



######### OUR CORRECTION #########

def Chloride_Correction2(df, df2):


    ######### RAIN DATA APPENDING #########
    
    rain = pd.read_excel('chemweathering/agilentexporttools/Rain_Ions.xlsx', sheet_name='Conconcentration (Processed)')
    
    rain = rain.loc[rain['Label'] == 'Refugio V. Rainwater'] # so far the best charge balanced one
    
    # Convert concentrations to molar units (assuming molar_conc_seawater function is defined)
    rain_processed = molar_conc_seawater(rain)
    
    ##### CHANGE HERE ABOVE FOR MORE SOPHISTICATED RAIN CHARGE BALANCE ANALYSIS #####
    
    ## Change the first column, first row name to 'unique_code'
    rain_processed.columns = ['unique_code'] + [col for col in rain_processed.columns[1:]]
    
    ########################################
    
    
    
    ######### RAIN CHARGE BALANCE #########
    
    # Charge balance
    rain_processed = charge_balance_modded(rain_processed)
    
    # Initialize lists to store unique codes and NICB values 
    NICB_Balance = NICB_Valid(rain_processed)
    
    # Print here if you want the NICB breakdown for the rain data #
    
    
    
    ######### ADDING HCO3 TO COLUMN #########

    # Ensure columns are in correct order
    rain_processed = rain_processed.loc[:, ['Ca [aq] (mM)', 'Mg [aq] (mM)', 'K [aq] (mM)', 'Na [aq] (mM)', 'Cl [aq] (mM)', 'SO4 [aq] (mM)', 'Sr [aq] (mM)']]
    
    # ADD HCO3 [aq] (mM) to rain_processed for all rows
     
    #rain_processed['HCO3 [aq] (mM)'] = 3.87 * rain_processed['Cl [aq] (mM)'] # AVERAGE ESTIMATED
    
    #rain_processed['HCO3 [aq] (mM)'] = 2.56 * rain_processed['Cl [aq] (mM)']
    
    #rain_processed['HCO3 [aq] (mM)'] = 5.17 * rain_processed['Cl [aq] (mM)']
    
    rain_processed['HCO3 [aq] (mM)'] = 0.004029989 * rain_processed['Cl [aq] (mM)'] # SEAWATER VALUE
     
    #############################################
    
    
    
    
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
    
    # Iterate over each element (column) in the rain_processed DataFrame
    for element in rain_processed.columns:
    # Check if the current element exists as a column in the df_copy DataFrame
    
        if element in df_copy.columns:
        # Get the value of the current element from the rain_processed DataFrame (assuming there's only one value)
        
            rain_element_value = rain_processed[element].values[0]
        
        # Get the value of 'Cl [aq] (mM)' from the rain_processed DataFrame (assuming there's only one value)
        
            cl_rain_value = rain_processed['Cl [aq] (mM)'].values[0]
            
            # Perform the correction calculation:
            # For each row in df_copy, subtract the rain element value from the river element value
            
            df_copy['*' + element] = df_copy[element] - rain_element_value 


        else:
        # Print a message if the current element is not found in the df_copy DataFrame
            print(f"Element {element} not found in df_copy")
    
    
    #################################################################################
    
    
    
    ######### EVAPORITE ENDMEMBER RATIO CALCULATION #########
    
    # For samples with water body type 'spring', calculate all element/Cl ratios for the corrected "*" values and store them in a dictionary:
    spring_ratios = {}

    # Get the row for the spring sample
    #spring_row = df_copy[df_copy['unique_code'] == 'T1AWF-1121'] 
    
    ######### PICKING COWLICK DEATH ROCK #########
    
    spring_row = df_copy[df_copy['unique_code'] == 'T1DW-0322']
    
    
    ## Print the spring row:
    #print(spring_row)
    # Iterate over each element in rain_processed
    for element in rain_processed.columns:
        if '*' + element in df_copy.columns and element != '*Cl [aq] (mM)':
            # Calculate the ratio for the current element
            ratio = spring_row['*' + element].values[0] / spring_row['*Cl [aq] (mM)'].values[0]
            # Store the ratio in the spring_ratios dictionary
            spring_ratios[element] = ratio

    # Calculate the mean ratio for each element from the spring sample
    mean_spring_ratios = {element: ratio for element, ratio in spring_ratios.items()}
    
    # Iterate over each element in mean_spring_ratios
    for element, ratio in mean_spring_ratios.items():
        # Divide the ratio by the initial Na/Cl ratio
        mean_spring_ratios[element] /= mean_spring_ratios['Na [aq] (mM)']
    
    #print(mean_spring_ratios)


    ################################################################
    


    ######### CORRECT USING THE SPRING RATIOS #########
        
    
    ## The spring ratios are now the ratios we will use to correct once more:
    
    
    # Apply the spring ratios to correct the elements for the remaining Cl amount
    for element in rain_processed.columns:
        if '*' + element in df_copy.columns:
            # Correct the element values using the spring ratios
            for index, row in df_copy.iterrows():
                if row['water_body'] != 'spring':
                    ratio = mean_spring_ratios[element]
                    df_copy['+' + element] = df_copy['*' + element] - (ratio * df_copy['*Cl [aq] (mM)'])
        else:
            print(f"Corrected element *{element} not found in df_copy")

    # Output the corrected DataFrame
    #print(df_copy)
    
    ######################################################
    
    
    
    
    
    
    
    ################## PLOTTING NON-EVAPORITE CORRECTED RESULTS ##################
            
    ## make a plot of the Cl* values in df_copy:
    for col in df_copy.columns:
        if '+' in col:
            plt.figure(figsize=(10,6))
            plt.scatter(df_copy['altitude_in_m'], df_copy[col], alpha=0.7, s=70, label=col)
            plt.ylabel('Concentration (mM)')
            plt.xlabel('Altitude (m)')
            plt.title(f'Scatter plot of {col} vs. Altitude - Evaporites Corrected')
            plt.legend()
            # plt.savefig(f'chemweathering/figures/{col}.png')
            
            for index, row in df_copy.iterrows():
                plt.text(row['altitude_in_m'], row[col], row['unique_code'], fontsize=8, ha='center', va='bottom')
            
            #plt.show()
            plt.close()
            
    ## Make a list of all the unique codes of the samples that have elements with negative values:
    
    neg_values = []

    for col in df_copy.columns:
        if '*' in col:
            if df_copy[col].min() < 0:
                neg_entries = df_copy.loc[df_copy[col] < 0, ['unique_code', col]]
                for _, row in neg_entries.iterrows():
                    neg_values.append([row['unique_code'], col, row[col]])

    # Create a DataFrame from the negative values list
    neg_df = pd.DataFrame(neg_values, columns=['unique_code', 'column_name', 'negative_value'])

    # Save the DataFrame to a CSV file
    neg_df.to_csv('negative_values.csv', index=False)
    
    
    
    
    
    ######### CREATING THE DF_MOCK DATAFRAME #########
    
    # df_mock returns the evaporite corrected
    df_mock = df_copy[['unique_code'] + [col for col in df_copy.columns if '+' in col] + ['latitude_converted', 'longitude_converted', 'NICB', 'calculated_discharge_in_m3_s-1', 'Alkalinity_CaCO3_in_mg_kg-1', 'field_pH', 'altitude_in_m', 'water_body']]

    # Select relevant columns for final output. df_copy returns the normal rain correction
    df_copy = df_copy[['unique_code'] + [col for col in df_copy.columns if '*' in col] + ['latitude_converted', 'longitude_converted', 'NICB', 'calculated_discharge_in_m3_s-1', 'Alkalinity_CaCO3_in_mg_kg-1', 'field_pH', 'altitude_in_m', 'water_body']]

    
    
    ######### PRINTING WHAT VALUES ARE NEGATIVE #########
    

    # remove all values that are negative in df_mock that have a "+" in the column name 
    unique_codes = set()
    for col in df_mock.columns:
        if '+' in col:
            negative_samples = df_mock[df_mock[col] < 0]
            #print(f"Negative samples for column {col}:")
            #print(negative_samples)
            unique_codes.update(negative_samples['unique_code'])
            df_mock = df_mock[df_mock[col] >= 0]
    
    print("Unique codes with negative values:")
    print(unique_codes)
    
    ######################################################
    
    
    
    #print(df_mock.columns)
    
    
    ######### PRUNING THE DATAFRAME #########
    
    
    # df_mock = df_mock[~df_mock['unique_code'].str.contains('T1DW-0322', na=False)]
    # df_mock = df_mock[~df_mock['unique_code'].str.contains('RC15WF-1121', na=False)] 
    # df_mock = df_mock[~df_mock['unique_code'].str.contains('T1BWF-1121', na=False)]
    # df_mock = df_mock[~df_mock['unique_code'].str.contains('T2AW-0322', na=False)]
    # df_mock = df_mock[~df_mock['unique_code'].str.contains('RC02AWF-0322', na=False)]
    # df_mock = df_mock[~df_mock['unique_code'].str.contains('T2a-0622', na=False)]
    # df_mock = df_mock[~df_mock['unique_code'].str.contains('T2AWF-1121', na=False)]
    
    
    
    ################## BLATTMANN WEATHERING PLOT ##################
    
    ##### Plot +SO4/(sum of +Ca, +Mg, 2 times +K, 2 times +Na) vs +HCO3/(sum of +Ca, +Mg, 2 times +K, 2 times +Na)
    plt.figure(figsize=(10,6))

    numerator = df_mock['+SO4 [aq] (mM)']
    denominator = df_mock['+Ca [aq] (mM)'] + df_mock['+Mg [aq] (mM)'] + df_mock['+K [aq] (mM)']/2 + df_mock['+Na [aq] (mM)']/2

    # Calculate the x and y values for the scatter plot
    x_values = df_mock['+HCO3 [aq] (mM)'] / denominator
    y_values = numerator / denominator

    plt.scatter(x_values, y_values, alpha=0.7, s=70)

    plt.xlabel('+HCO3/(sum of +Ca, +Mg, +K/2, +Na/2)')
    plt.ylabel('+SO4/(sum of +Ca, +Mg, +K/2, +Na/2)')
    # Add text annotations
    for i in range(len(df_mock)):
        plt.text(x_values.iloc[i], y_values.iloc[i], df_mock['unique_code'].iloc[i], fontsize=8, ha='center', va='bottom')

    # Calculate the line of best fit
    x = pd.to_numeric(x_values, errors='coerce')
    y = pd.to_numeric(y_values, errors='coerce')

    # Combine x and y into a DataFrame and drop rows with NaN values
    data = pd.DataFrame({'x': x, 'y': y}).dropna()

    # Extract the cleaned x and y values
    x_clean = data['x']
    y_clean = data['y']

    # Add a constant to the independent variable
    X = sm.add_constant(x_clean)

    # Fit the OLS model
    model = sm.OLS(y_clean, X).fit()

    # Make predictions
    predictions = model.predict(X)

    # Plot the line of best fit
    plt.plot(x_clean, predictions, color='blue')

    # Add gradient and intercept to the label
    gradient = round(model.params.iloc[1], 2)
    intercept = round(model.params.iloc[0], 2)
    label = f'Our line of best fit (y = {gradient}x + {intercept})'
    plt.plot([], [], c='blue', linewidth=2, label=label)

    # Plot the stoichiometric line y = -1/2x + 1
    x_line = np.linspace(0, 2, 100)
    y_line = -1/2 * x_line + 1
    plt.plot(x_line, y_line, color='red')

    # Add legend
    plt.plot([], [], c='red', linewidth=2, label='Stoichiometric line y = -1/2x + 1')
    #plt.plot([], [], c='blue', linewidth=2, label='Our line of best fit')
    plt.legend()

    plt.title('Plot of +SO4/(sum of +Ca, +Mg, +K/2, +Na/2) vs +HCO3/(sum of +Ca, +Mg, +K/2 , +Na/2) \n ESTIMATED SEAWATER HCO3')
    #plt.show()
    
    plt.close()
        
    
    #############################################
    
    
    ######### RETURNING THE RELEVANT DATAFRAMES #########



    na_after = df_mock['+Na [aq] (mM)']
    
    #print('Na percentage:')
    
    na_percentage = na_after / na_before
    
    #print(na_percentage)
    
    ## Sensitive to corrections. Big uncertainty
    
    
    #############################################
    
    
    #############################################
    
    ######### PLOTTING FOR SECONDARY CALCITE INVESTIGATION #########
    
    
    
    
    
    # Plot Mg/Ca on the Y against Na/Ca on the X
    
    df_mock['Sr/Ca'] = df_mock['+Sr [aq] (mM)'] / df_mock['+Ca [aq] (mM)']
    
    df_mock['Mg/Ca'] = df_mock['+Mg [aq] (mM)'] / df_mock['+Ca [aq] (mM)']
    
    df_mock['Na/Ca'] = df_mock['+Na [aq] (mM)'] / df_mock['+Ca [aq] (mM)']
    
    
    plt.figure(figsize=(10,6))
    
    # Plot the scatter points with different markers for each water body type
    markers = {'mainstream': 'o', 'tributary': '^', 'spring': '*'}

    for water_body, marker in markers.items():
        subset = df_mock[df_mock['water_body'] == water_body]
        sc = plt.scatter(
            subset['Na/Ca'].values,
            subset['Sr/Ca'].values,
            c='black',
            s=100,
            alpha=0.7,
            edgecolor='black',
            marker=marker,
            label=water_body,
            linewidth=1
        )

    # Create legend with custom markers and labels
    for water_body, marker in markers.items():
        plt.scatter([], [], marker=marker, color='black', label=water_body)

    plt.ylabel('Sr/Ca')
    plt.xlabel('Na/Ca')
    plt.legend()
    
    for index, row in df_mock.iterrows():
        plt.text(row['Na/Ca'], row['Sr/Ca'], row['unique_code'], fontsize=8, ha='center', va='bottom')
        
    plt.title('Plot of Sr/Ca vs Na/Ca')
    
    #plt.show()
    plt.close()

    
    
    plt.figure(figsize=(10,6))
    plt.scatter(df_mock['Na/Ca'], df_mock['Mg/Ca'], alpha=0.7, s=70)
    plt.ylabel('Mg/Ca')
    plt.xlabel('Na/Ca')
    
    for index, row in df_mock.iterrows():
        plt.text(row['Na/Ca'], row['Mg/Ca'], row['unique_code'], fontsize=8, ha='center', va='bottom')
        
    
    plt.title('Plot of Mg/Ca vs Na/Ca')
    
    #plt.show()
    plt.close()
    
    
    
    
    plt.figure(figsize=(10,6))
    plt.scatter(df_mock['Sr/Ca'], df_mock['+Sr [aq] (mM)'], alpha=0.7, s=70)
    plt.ylabel('+Sr [aq] (mM)')
    plt.xlabel('Sr/Ca')
    
    for index, row in df_mock.iterrows():
        plt.text(row['Sr/Ca'], row['+Sr [aq] (mM)'], row['unique_code'], fontsize=8, ha='center', va='bottom')
        
    
    plt.title('Plot of Sr vs Sr/Ca')
    
    #plt.show()
    plt.close()
    
    
    
     
    plt.figure(figsize=(10,6))
    plt.scatter(df_mock['Sr/Ca'], df_mock['+Ca [aq] (mM)'], alpha=0.7, s=70)
    plt.ylabel('+Ca [aq] (mM)')
    plt.xlabel('Sr/Ca')
    
    for index, row in df_mock.iterrows():
        plt.text(row['Sr/Ca'], row['+Ca [aq] (mM)'], row['unique_code'], fontsize=8, ha='center', va='bottom')
        
    
    plt.title('Plot of Ca vs Sr/Ca')
    
    #plt.show()
    plt.close()
    
    


    return df_copy, df_mock

