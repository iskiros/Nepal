import sys
import os
import math
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





def fraction_calculation(df):
    
    df_copy = df.copy()
    
    # Firstly, calculate SO4 2- total carb
    # Assuming SO4 2- pyrite is the only source of sulfate
    
    CaNa_sil = 0.59
    
    MgNa_sil = 0.36
    
    
    
    
    ######################################################
    ####### RELPH THESIS WAY #######
    ######################################################
    
    
    # df_copy['Ca_Sil'] = df_copy['+Na [aq] (mM)'] * CaNa_sil
    
    # df_copy['Mg_Sil'] = df_copy['+Na [aq] (mM)'] * MgNa_sil
    
    # ### Get someone to double check this is right
    
    # df_copy['Ca_Carb'] = df_copy['+Ca [aq] (mM)'] - df_copy['Ca_Sil']
    
    # df_copy['Mg_Carb'] = df_copy['+Mg [aq] (mM)'] - df_copy['Mg_Sil']
    
    
    
    
    # df_copy['SO4 2- total carb'] = df_copy['+SO4 [aq] (mM)']*(df_copy['Mg_Carb'] + df_copy['Ca_Carb'])/(df_copy['Ca_Sil'] + df_copy['Mg_Sil'] + df_copy['Mg_Carb'] + df_copy['Ca_Carb'] + df_copy['+K [aq] (mM)'] + df_copy['+Na [aq] (mM)'])
    
    # df_copy['SO4 2- total carb'] = df_copy['SO4 2- total carb'].apply(lambda x: x if x >= 0 else 0)
    
    # #print(df_copy['SO4 2- total carb'])
    
    
    
    
    # df_copy['SO4 2- total sil'] = df_copy['+SO4 [aq] (mM)']*(df_copy['Ca_Sil'] + df_copy['Mg_Sil'] + df_copy['+K [aq] (mM)'] + df_copy['+Na [aq] (mM)'])/(df_copy['Ca_Sil'] + df_copy['Mg_Sil'] + df_copy['Mg_Carb'] + df_copy['Ca_Carb'] + df_copy['+K [aq] (mM)'] + df_copy['+Na [aq] (mM)'])
    
    # #print(df_copy['SO4 2- total sil'])
    


    # df_copy['Ca sulfO carb'] = df_copy['SO4 2- total carb']*(df_copy['Ca_Carb']/(df_copy['Ca_Carb'] + df_copy['Mg_Carb']))
    
    # df_copy['Ca sulfC carb'] = 2*df_copy['SO4 2- total carb']*(df_copy['Ca_Carb']/(df_copy['Ca_Carb'] + df_copy['Mg_Carb']))
    
    # df_copy['Mg sulfO carb'] = df_copy['SO4 2- total carb']*(df_copy['Mg_Carb']/(df_copy['Ca_Carb'] + df_copy['Mg_Carb']))
    
    # df_copy['Mg sulfC carb'] = 2*df_copy['SO4 2- total carb']*(df_copy['Mg_Carb']/(df_copy['Ca_Carb'] + df_copy['Mg_Carb']))
    
    
    # df_copy['Ca sulf sil'] = df_copy['SO4 2- total sil']*(df_copy['Ca_Sil']/(df_copy['Ca_Sil'] + df_copy['Mg_Sil'] + df_copy['+K [aq] (mM)'] + df_copy['+Na [aq] (mM)']))
    
    # df_copy['Mg sulf sil'] = df_copy['SO4 2- total sil']*(df_copy['Mg_Sil']/(df_copy['Ca_Sil'] + df_copy['Mg_Sil'] + df_copy['+K [aq] (mM)'] + df_copy['+Na [aq] (mM)']))
    
    # df_copy['Na sulf sil']= df_copy['SO4 2- total sil']*(df_copy['+Na [aq] (mM)']/(df_copy['Ca_Sil'] + df_copy['Mg_Sil'] + df_copy['+K [aq] (mM)'] + df_copy['+Na [aq] (mM)']))
    
    # df_copy['K sulf sil'] = df_copy['SO4 2- total sil']*(df_copy['+K [aq] (mM)']/(df_copy['Ca_Sil'] + df_copy['Mg_Sil'] + df_copy['+K [aq] (mM)'] + df_copy['+Na [aq] (mM)']))
    
    
    
    # df_copy['Ca CarbO Carb'] = df_copy['+Ca [aq] (mM)'] - df_copy['Ca sulfO carb']
    
    # df_copy['Ca CarbC Carb'] = df_copy['+Ca [aq] (mM)'] - df_copy['Ca sulfC carb']
    
    # df_copy['Mg CarbO Carb'] = df_copy['+Mg [aq] (mM)'] - df_copy['Mg sulfO carb']
    
    # df_copy['Mg CarbC Carb'] = df_copy['+Mg [aq] (mM)'] - df_copy['Mg sulfC carb']
    
    
    
    # df_copy['Ca Carb Sil'] = df_copy['+Ca [aq] (mM)'] - df_copy['Ca sulf sil']
    
    # df_copy['Mg Carb Sil'] = df_copy['+Mg [aq] (mM)'] - df_copy['Mg sulf sil']
    
    # df_copy['Na Carb Sil'] = df_copy['+Na [aq] (mM)'] - df_copy['Na sulf sil']
    
    # df_copy['K Carb Sil'] = df_copy['+K [aq] (mM)'] - df_copy['K sulf sil']
    
    
    
    
    # # Relph's thesis is not very clear on this so I just assume you add the open and closed
    # df_copy['Ca Carb Carb'] = df_copy['Ca CarbO Carb'] + df_copy['Ca CarbC Carb']
    
    
    
    # df_copy['Fcarb'] = (df_copy['Ca Carb Carb'] + df_copy['Ca sulfO carb'])/(df_copy['Ca Carb Sil'] + df_copy['Ca Carb Carb'] + df_copy['Ca sulfO carb'] + df_copy['Ca sulf sil'])
    
    # df_copy['Fsulf'] = (df_copy['Ca sulf sil'] + df_copy['Ca sulfO carb'])/(df_copy['Ca Carb Sil'] + df_copy['Ca Carb Carb'] + df_copy['Ca sulfO carb'] + df_copy['Ca sulf sil'])
    
    
    
    # df_copy['Fcarb_closed'] = (2*df_copy['Ca Carb Carb'] + df_copy['Ca sulfC carb'])/(2*df_copy['Ca Carb Sil'] + df_copy['Ca Carb Carb'] + df_copy['Ca sulfC carb'] + df_copy['Ca sulf sil'])
    
    # df_copy['Fsulf_closed'] = (df_copy['Ca sulf sil'] + 2*df_copy['Ca sulfC carb'])/(2*df_copy['Ca Carb Sil'] + df_copy['Ca Carb Carb'] + df_copy['Ca sulfC carb'] + df_copy['Ca sulf sil'])
    
    
    
    
    ################################################################
    ################################################################
    
    
    
    
    
    ######################################################
    ################## SIMPLE WAY ##################
    ######################################################
    
    df_copy['Fsulf_simple'] = 2*df_copy['+SO4 [aq] (mM)']/(2*df_copy['+SO4 [aq] (mM)'] + df_copy['+HCO3 [aq] (mM)'])
    
    df_copy['Fcarb_simple'] = 1-df_copy['X_Sil']
    
    ###################################
    
    
    
    
    ######### FCARB FSULF PLOT #########
    
    ## Export dataframes to csv files:
    #df_copy.to_csv('chemweathering/data/df_copy for FCarb.csv', index=False)
    
    ## Plot F sulf on the y and F carb on the x. Add limits of 0 and 1 for both axes to the plot and make the plot square
    
    # Define the gradient size (resolution)
    gradient_size = 100

    # Create a gradient from orange to cyan
    # Orange color: (1, 0.5, 0), Cyan color: (0, 1, 1)
    gradient = np.zeros((gradient_size, gradient_size, 4))  # Use 4 channels for RGBA

    for i in range(gradient_size):
        for j in range(gradient_size):
            t = (i + j) / (2 * gradient_size)  # Normalized distance
            # Linear interpolation between orange and cyan, with lighter alpha
            gradient[i, j] = (1 - t, 0.2 * (1 - 0.5*t) + t, t, 0.8)  # Adjust alpha for lighter colors

    # Clip the gradient values to the valid range for imshow
    gradient = np.clip(gradient, 0, 1)

    
    plt.figure(figsize=(8, 8))
    
    
    
    # Define markers for each water body type
    markers = {'mainstream': 'o', 'tributary': '^', 'spring': '*'}
    
    
    
    ################## IF WERE TO PLOT RELPH's VALUES ##################

    # # Plot the scatter points with different markers for each water body type
    # for water_body, marker in markers.items():
    #     subset = df_copy[df_copy['water_body'] == water_body]
    #     plt.scatter(
    #         subset['Fcarb'],
    #         subset['Fsulf'],
    #         c='black',  # Use a constant color or map based on another variable
    #         s=100,
    #         alpha=0.9,
    #         edgecolor='white',
    #         marker=marker,
    #         label=water_body,
    #         linewidth=1,
    #         zorder = 2
    #     )


    # # Plot the scatter points with different markers for each water body type
    # for water_body, marker in markers.items():
    #     subset = df_copy[df_copy['water_body'] == water_body]
    #     plt.scatter(
    #         subset['Fcarb_closed'],
    #         subset['Fsulf_closed'],
    #         c='blue',  # Use a constant color or map based on another variable
    #         s=100,
    #         alpha=0.9,
    #         edgecolor='white',
    #         marker=marker,
    #         linewidth=1,
    #         zorder = 2
    #     )


    # ## Fudge a blue point label":
    # plt.scatter([], [], c='blue', label='Closed System', edgecolor='white', s=100, zorder = 2)
    
    # ## Fudge a blue point label":
    # plt.scatter([], [], c='black', label='Open System', edgecolor='white', s=100, zorder = 2)
    
    
    # for index, row in df_copy.iterrows():
    #     plt.text(row['Fcarb'], row['Fsulf'], row['unique_code'][:4], fontsize=6)
    
    # for index, row in df_copy.iterrows():
    #     plt.text(row['Fcarb_closed'], row['Fsulf_closed'], row['unique_code'][:4], fontsize=6)
    
    ######################################################
    ######################################################
    
    
    
    
    
    
    ################## SIMPLE VALUES ##################
    ######################################################
    
    # Plot the scatter points with different markers for each water body type
    for water_body, marker in markers.items():
        subset = df_copy[df_copy['water_body'] == water_body]
        plt.scatter(
            subset['Fcarb_simple'],
            subset['Fsulf_simple'],
            c='black',  # Use a constant color or map based on another variable
            s=100,
            alpha=0.9,
            edgecolor='white',
            marker=marker,
            label=water_body,
            linewidth=1,
            zorder = 2
        )
    
    for index, row in df_copy.iterrows():
        plt.text(row['Fcarb_simple'], row['Fsulf_simple'], row['unique_code'][:4], fontsize=6)

    
    
    # Display the gradient as an image
    plt.imshow(gradient, extent=[0, 1, 0, 1], origin='lower', aspect='auto')
    
    
    # Add a line at y = -x + 1
    x = np.linspace(0, 1, 100)
    y1 = -x + 1
    plt.plot(x, y1, color='black', linestyle='--')
    
    # Shade the bit with y above this line light gray:
    plt.fill_between(x, y1, 1, color='lightgray', alpha = 0.5)

    # Add another line at y = 1 - (0.5x)
    y2 = 1 - (0.5 * x)
    plt.plot(x, y2, color='black', linestyle='--')
    # Shade the bit with y above this line light gray:
    plt.fill_between(x, y2, 1, color='gray', alpha = 0.5)
    
    
    plt.text(0.65, 0.4, 'CO2 Release long term', fontsize=12, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', zorder = 2))
    plt.text(0.65, 0.8, 'CO2 release short term', fontsize=12, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', zorder = 2))

    # Add a legend
    plt.legend()

    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    
    ## Add a gradient going from red to blue to the plot. Red in the bottom left and blue in the top right
    
    
    plt.xlabel('Fcarb')
    plt.ylabel('Fsulf')
    plt.title('Fraction of Carb vs. Sulf')
    #plt.show()
    
    plt.close()
    
    

    #print(df_copy['gradient_color'])
    
    
    #############################################################
    #############################################################
    
    
    
    ################## CALCULATING X VALUES ##################
    
    df_copy['X_Carb_Sulf'] = df_copy['Fcarb_simple'] * df_copy['Fsulf_simple']
    
    df_copy['X_Carb_Carb'] = df_copy['Fcarb_simple'] * (1 - df_copy['Fsulf_simple'])
    
    df_copy['X_Sil_Sulf'] = df_copy['X_Sil'] * df_copy['Fsulf_simple']
    
    df_copy['X_Sil_Carb'] = df_copy['X_Sil'] * (1 - df_copy['Fsulf_simple'])
    
    
    
    #export to csv
    #df_copy.to_csv('chemweathering/data/df_copy for X_Carb_Sulf.csv', index=False)
    
    ######################################################
    ######################################################
    
    df_copy['HCO3_Tot'] = 0
    
    element_dict = {
        'molecules': ['+Ca', '+Mg', '+K', '+Na', '+HCO3', '+Cl', '+SO4', '+Al', '+As', '+Ba', '+Bi', '+Ce', '+Fe', '+Li', '+Rb', '+Sr', '+CO3', '+H2PO4'],
        'molar_mass': [40.08, 24.31, 39.10, 22.99, 61.02, 35.45, 96.06, 26.98, 74.92, 137.33, 208.98, 140.12, 55.85, 6.94, 85.47, 87.62, 60.01, 97.99],
        'valency': [2, 2, 1, 1, 1, 1, -2, 3, 3, 2, 3, 3, 2, 1, 1, 2, 2, 1]
    }

    for element, molar_mass, valency in zip(element_dict['molecules'],
                                            element_dict['molar_mass'],
                                            element_dict['valency']):
        column_name = element + ' [aq] (mM)'
        if column_name in df_copy.columns:
            df_copy['HCO3_Tot'] += df_copy[column_name] * valency
    
    
    ######################################################
    ######## ######## ######## ######## ######## ######## 
    
    
    
    
    
    ######################################################
    ######## TESTING OUT OUR HCO3- CALCULATIONS ########
    ######################################################
    
    plt.figure(figsize=(8, 8))
    # Plot of HCO3_Tot against +HCO3 [aq] (mM)
    
    # add a y = x lineß
    
    #print(round(df_copy['HCO3_Tot'].max()))
    
    ################## SEEING WHAT IS BETTER, HCO3_TOT OR OUR MEASURED HCO3 ##################
    
    
    x = np.linspace(0, 12, 100)
    y = x
    plt.plot(x, y, color='black', linestyle='--', label='y = x')
    
    plt.scatter(df_copy['HCO3_Tot'], df_copy['+HCO3 [aq] (mM)'])
    plt.xlabel('HCO3_Tot')
    plt.ylabel('+HCO3 [aq] (mM)')
    plt.title('HCO3_Tot vs. +HCO3 [aq] (mM)')
    
    # Calculate the line of best fit
    slope, intercept = np.polyfit(pd.to_numeric(df_copy['HCO3_Tot'], errors='coerce'), pd.to_numeric(df_copy['+HCO3 [aq] (mM)'], errors='coerce'), 1)
    line = slope * x + intercept
    
    # Plot the line of best fit
    plt.plot(x, line, color='red', label='Line of Best Fit')
    
    # Display the equation of the line of best fit
    equation = f'y = {slope:.2f}x + {intercept:.2f}'
    plt.text(2, 10, equation, fontsize=12, color='red')
    
    plt.legend()
    #plt.show()
    
    plt.close()
    
    
    
    ######## ######## ######## ######## ######## ######## 
    ######################################################
    ######################################################
    
    
    
    # Convert alkalinity to meq/L
    df_copy['Alkalinity (meq/L)'] = df_copy['Alkalinity_CaCO3_in_mg_kg-1'] / 50.05
    
    # Plot alkalinity in meq/L against +HCO3 [aq] (mM)
    
    plt.figure(figsize=(8, 8))
    
    x = np.linspace(0, 12, 100)
    y = x
    plt.plot(x, y, color='black', linestyle='--', label='y = x')
    
    plt.scatter(df_copy['Alkalinity (meq/L)'], df_copy['+HCO3 [aq] (mM)'])
    plt.xlabel('Alkalinity (meq/L)')
    plt.ylabel('+HCO3 [aq] (mM)')

    plt.title('Alkalinity (meq/L) vs. +HCO3 [aq] (mM)')
    
    # Calculate the line of best fit
    slope, intercept = np.polyfit(pd.to_numeric(df_copy['Alkalinity (meq/L)'], errors='coerce'), pd.to_numeric(df_copy['+HCO3 [aq] (mM)'], errors='coerce'), 1)
    line = slope * x + intercept
    
    # Plot the line of best fit
    plt.plot(x, line, color='red', label='Line of Best Fit')
    
    # Display the equation of the line of best fit
    equation = f'y = {slope:.2f}x + {intercept:.2f}'
    plt.text(2, 10, equation, fontsize=12, color='red')
    
    plt.legend()
    #plt.show()
    
    plt.close()   
    
    
    ######## ######## ######## ######## ######## ######## 
    ######################################################
    ######## ######## ######## ######## ######## ######## 
    
    
    
        
    ### Previously used HCO3_Tot to calculate HCO3_Sil_Sulf, HCO3_Carb_Carb, HCO3_Carb_Sulf, HCO3_Sil_Carb.
    ### Has since then been changed because HCO3_Tot was not accurate. See log. Now using +HCO3 [aq] (mM) to calculate these values.    
        
    # df_copy['HCO3_Sil_Sulf'] = df_copy['HCO3_Tot'] * df_copy['X_Sil_Sulf']
    
    # df_copy['HCO3_Carb_Carb'] = df_copy['HCO3_Tot'] * df_copy['X_Carb_Carb']
    
    # df_copy['HCO3_Carb_Sulf'] = df_copy['HCO3_Tot'] * df_copy['X_Carb_Sulf']
    
    # df_copy['HCO3_Sil_Carb'] = df_copy['HCO3_Tot'] * df_copy['X_Sil_Carb']
    
    
        
    df_copy['HCO3_Sil_Sulf'] = df_copy['+HCO3 [aq] (mM)'] * df_copy['X_Sil_Sulf']
    
    df_copy['HCO3_Carb_Carb'] = df_copy['+HCO3 [aq] (mM)'] * df_copy['X_Carb_Carb']
    
    df_copy['HCO3_Carb_Sulf'] = df_copy['+HCO3 [aq] (mM)'] * df_copy['X_Carb_Sulf']
    
    df_copy['HCO3_Sil_Carb'] = df_copy['+HCO3 [aq] (mM)'] * df_copy['X_Sil_Carb']
    
    
    ######################################################
    ###################################################### 
    ######## DISCHARGE CALCULATION AND EXPORT ########
    
    # To achieve the goal of adding the maximum discharge for rivers or tributaries when there are multiple estimations, 
    # we'll follow these steps:

    # Filter the DataFrame to separate out the rivers (beginning with 'R') and tributaries (beginning with 'T').
    # Group by the first few characters that identify the river or tributary.
    # Aggregate the discharge values by taking the maximum for each group.
    # Merge these maximum values back into the original DataFrame.
        
    # Filter the dataframe to include only mainstream samples (rivers) and explicitly make a copy
    df_mainstream = df_copy[df_copy['unique_code'].str.startswith('R')].copy()
    df_mainstream['unique_code_trimmed'] = df_mainstream['unique_code'].str[:4]

    # Calculate the maximum discharge for each river
    max_discharge_mainstream = df_mainstream.groupby('unique_code_trimmed')['calculated_discharge_in_m3_s-1'].max().reset_index()
    max_discharge_mainstream.rename(columns={'calculated_discharge_in_m3_s-1': 'max_discharge_in_m3_s-1'}, inplace=True)

    # Merge the maximum discharge back to the mainstream dataframe
    df_mainstream = df_mainstream.merge(max_discharge_mainstream, on='unique_code_trimmed', how='left')
    df_mainstream['calculated_discharge_in_m3_s-1'] = df_mainstream['max_discharge_in_m3_s-1']
    df_mainstream.drop(columns=['max_discharge_in_m3_s-1'], inplace=True)

    # Filter the dataframe to include only tributary samples and explicitly make a copy
    df_trib = df_copy[df_copy['unique_code'].str.startswith('T')].copy()
    df_trib['unique_code_trimmed'] = df_trib['unique_code'].str[:3]
    df_trib['unique_code_trimmed'] = df_trib['unique_code_trimmed'].str.extract('T(\d+)')[0].astype(int)  # Extract the numeric part

    # Calculate the maximum discharge for each tributary
    max_discharge_tributary = df_trib.groupby('unique_code_trimmed')['calculated_discharge_in_m3_s-1'].max().reset_index()
    max_discharge_tributary.rename(columns={'calculated_discharge_in_m3_s-1': 'max_discharge_in_m3_s-1'}, inplace=True)

    # Merge the maximum discharge back to the tributary dataframe
    df_trib = df_trib.merge(max_discharge_tributary, on='unique_code_trimmed', how='left')
    df_trib['calculated_discharge_in_m3_s-1'] = df_trib['max_discharge_in_m3_s-1']
    df_trib.drop(columns=['max_discharge_in_m3_s-1'], inplace=True)

    # Now, merge the mainstream and tributary dataframes back into df_copy
    # Merge for mainstream (Rivers)
    df_copy = df_copy.merge(df_mainstream[['unique_code', 'calculated_discharge_in_m3_s-1']], on='unique_code', how='left', suffixes=('', '_max'))
    df_copy['calculated_discharge_in_m3_s-1'] = df_copy['calculated_discharge_in_m3_s-1_max'].combine_first(df_copy['calculated_discharge_in_m3_s-1'])
    df_copy.drop(columns=['calculated_discharge_in_m3_s-1_max'], inplace=True)

    # Merge for tributaries
    df_copy = df_copy.merge(df_trib[['unique_code', 'calculated_discharge_in_m3_s-1']], on='unique_code', how='left', suffixes=('', '_max'))
    df_copy['calculated_discharge_in_m3_s-1'] = df_copy['calculated_discharge_in_m3_s-1_max'].combine_first(df_copy['calculated_discharge_in_m3_s-1'])
    df_copy.drop(columns=['calculated_discharge_in_m3_s-1_max'], inplace=True)

        
    
    ############################################################################################################
    ############################################################################################################
    
        
    df_copy['HCO3_Sil_Sulf_discharge'] = df_copy['HCO3_Sil_Sulf'] * df_copy['calculated_discharge_in_m3_s-1']
    df_copy['HCO3_Carb_Carb_discharge'] = df_copy['HCO3_Carb_Carb'] * df_copy['calculated_discharge_in_m3_s-1']
    df_copy['HCO3_Carb_Sulf_discharge'] = df_copy['HCO3_Carb_Sulf'] * df_copy['calculated_discharge_in_m3_s-1']
    df_copy['HCO3_Sil_Carb_discharge'] = df_copy['HCO3_Sil_Carb'] * df_copy['calculated_discharge_in_m3_s-1']
    
    
    # Export to xlsx
    #df_copy.to_excel('chemweathering/data/df_copy for HCO3_Sil_Sulf.xlsx', index=False)
    
    df_copy['HCO3_Sil_Sulf_discharge_yr'] = df_copy['HCO3_Sil_Sulf_discharge'] * 60 * 60 * 24 * 365
    df_copy['HCO3_Carb_Carb_discharge_yr'] = df_copy['HCO3_Carb_Carb_discharge'] * 60 * 60 * 24 * 365
    df_copy['HCO3_Carb_Sulf_discharge_yr'] = df_copy['HCO3_Carb_Sulf_discharge'] * 60 * 60 * 24 * 365
    df_copy['HCO3_Sil_Carb_discharge_yr'] = df_copy['HCO3_Sil_Carb_discharge'] * 60 * 60 * 24 * 365
    
    
    ## The following in mol CO2 per year
    
    df_copy['Carbonate_Carbonic_Short_Term'] = -1*(df_copy['HCO3_Carb_Carb_discharge_yr'] * 0.5) # CO2 is consumed
    df_copy['Carbonate_Carbonic_Long_Term'] = (df_copy['HCO3_Carb_Carb_discharge_yr'] * 0.5) # CO2 is produced
    
    
    df_copy['Carbonate_Sulfuric_Short_Term'] = (df_copy['HCO3_Carb_Sulf_discharge_yr'] * 0.5) # CO2 is produced
    df_copy['Carbonate_Sulfuric_Long_Term'] = (df_copy['HCO3_Carb_Sulf_discharge_yr'] * 0.5)*2 # CO2 is produced
    
    
    df_copy['Silicate_Carbonic_Short_Term'] = -1*(df_copy['HCO3_Sil_Carb_discharge_yr']) # CO2 is consumed
    df_copy['Silicate_Carbonic_Long_Term'] = -1*(df_copy['HCO3_Sil_Carb_discharge_yr'] * 0.5) # CO2 is consumed
    
    
    df_copy['Silicate_Sulfuric_Short_Term'] = 0 # Nothing happens
    df_copy['Silicate_Sulfuric_Long_Term'] = 0 # Nothing happens
    
    
    
    # Convert to ktCO2/yr
    
    df_copy['Carbonate_Carbonic_Short_Term_Mass'] = df_copy['Carbonate_Carbonic_Short_Term'] * 44.01 * 10**-9
    df_copy['Carbonate_Carbonic_Long_Term_Mass'] = df_copy['Carbonate_Carbonic_Long_Term'] * 44.01 * 10**-9
    
    df_copy['Carbonate_Sulfuric_Short_Term_Mass'] = df_copy['Carbonate_Sulfuric_Short_Term'] * 44.01 * 10**-9
    df_copy['Carbonate_Sulfuric_Long_Term_Mass'] = df_copy['Carbonate_Sulfuric_Long_Term'] * 44.01 * 10**-9
    
    df_copy['Silicate_Carbonic_Short_Term_Mass'] = df_copy['Silicate_Carbonic_Short_Term'] * 44.01 * 10**-9
    df_copy['Silicate_Carbonic_Long_Term_Mass'] = df_copy['Silicate_Carbonic_Long_Term'] * 44.01 * 10**-9
    
    df_copy['Silicate_Sulfuric_Short_Term_Mass'] = df_copy['Silicate_Sulfuric_Short_Term'] * 44.01 * 10**-9
    df_copy['Silicate_Sulfuric_Long_Term_Mass'] = df_copy['Silicate_Sulfuric_Long_Term'] * 44.01 * 10**-9
    
    
    
    # Plot each weathering type and time on a separate tiny plot
    # Create a list of weathering types and times
    weathering_types = ['Carbonate_Carbonic', 'Carbonate_Sulfuric', 'Silicate_Carbonic'] # , 'Silicate_Sulfuric'] - commenting out silicate sulfuric because it is zero
    times = ['Short_Term', 'Long_Term']


    ######################################################
    ################## PLOTTING THE HISTOGRAMS ##################
    ######################################################


    ## Set the number of rows for the subplots
    num_rows = len(weathering_types) * len(times)

    # Create a figure with subplots with a larger height for each subplot
    fig, axes = plt.subplots(num_rows, 1, figsize=(12, num_rows * 2.5))
    
    num_bins = 20

    
    # Define the consistent x-axis limits
    x_limits = (-75, 75)

    # Iterate over each weathering type and time
    for i, weathering_type in enumerate(weathering_types):
        for j, time in enumerate(times):
            # Get the corresponding mass data
            mass_data = df_copy[f'{weathering_type}_{time}_Mass']
            
            # Plot the mass data on the corresponding subplot
            ax = axes[i * len(times) + j]
            ax.hist(mass_data, bins=num_bins, alpha=0.7)
            
            # Set the title and labels for the subplot
            ax.set_title(f'{weathering_type} {time}')
            ax.set_xlabel('CO2 Flux [ktCO2/yr]')
            ax.set_ylabel('Frequency of Observations')

            # Set the x-axis limit to be consistent across all subplots
            ax.set_xlim(x_limits)
            
             
            axes[2].text(40, -3.2, 'CO2 release to atmosphere', fontsize=12, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', zorder = 2))
            axes[2].text(-75, -3.2, 'Atmospheric CO2 uptake', fontsize=12, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', zorder = 2))

            
            
    # Adjust the spacing between subplots for better visibility
   
    
    plt.tight_layout()

    plt.savefig('chemweathering/data/Weathering_Types.png')
    # Show the plot
    #plt.show()
    
    
    
    plt.close()


    #############################################
    #############################################



    # Now add together the short term and long term budgets to make a total short term budget and a total long term budget, then plot them on a graph like we have done here
    
    # Calculate the total short-term budget
    df_copy['Total_Short_Term_Budget'] = (
        df_copy['Carbonate_Carbonic_Short_Term_Mass'] +
        df_copy['Carbonate_Sulfuric_Short_Term_Mass'] +
        df_copy['Silicate_Carbonic_Short_Term_Mass'] +
        df_copy['Silicate_Sulfuric_Short_Term_Mass']
    )

    # Calculate the total long-term budget
    df_copy['Total_Long_Term_Budget'] = (
        df_copy['Carbonate_Carbonic_Long_Term_Mass'] +
        df_copy['Carbonate_Sulfuric_Long_Term_Mass'] +
        df_copy['Silicate_Carbonic_Long_Term_Mass'] +
        df_copy['Silicate_Sulfuric_Long_Term_Mass']
    )

    # Calculate the net long-term budget
    #df_copy['Net_Long_Term_Budget'] = df_copy['Total_Long_Term_Budget'] + df_copy['Total_Short_Term_Budget']
    
    # Changing it so net long term budget is the same as the total long term budget
    df_copy['Net_Long_Term_Budget'] = df_copy['Total_Long_Term_Budget']


    # Create a figure for the total and net budgets
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))


    num_bins = 20


    # Define the consistent x-axis limits for total budgets
    x_limits_total = (-100, 100)

    # Plot the total short-term budget histogram
    axes[0].hist(df_copy['Total_Short_Term_Budget'], bins=num_bins, alpha=0.7)
    axes[0].set_title('Total Short Term Budget')
    axes[0].set_xlabel('CO2 Flux [ktCO2/yr]')
    axes[0].set_ylabel('Frequency of Observations')
    axes[0].set_xlim(x_limits_total)

    # Plot the total long-term budget histogram
    axes[1].hist(df_copy['Total_Long_Term_Budget'], bins=num_bins, alpha=0.7)
    axes[1].set_title('Total Long Term Budget')
    axes[1].set_xlabel('CO2 Flux [ktCO2/yr]')
    axes[1].set_ylabel('Frequency of Observations')
    axes[1].set_xlim(x_limits_total)

    # Plot the net long-term budget histogram
    axes[2].hist(df_copy['Net_Long_Term_Budget'], bins=num_bins, alpha=0.7)
    axes[2].set_title('Net Long Term Budget')
    axes[2].set_xlabel('CO2 Flux [ktCO2/yr]')
    axes[2].set_ylabel('Frequency of Observations')
    axes[2].set_xlim(-200, 200)  # Set custom limits for the net plot
    
    # Add labels to the plots
    axes[2].text(100, 6, 'CO2 release to atmosphere', fontsize=12, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', zorder = 2))
    axes[2].text(-150, 6, 'Atmospheric CO2 uptake', fontsize=12, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', zorder = 2))

    # Adjust the spacing beween subplots for better visibility
    plt.tight_layout()

    plt.savefig('chemweathering/data/Total_Net.png')
    # Show the plot
    #plt.show()
    
    plt.close()
    
    
    
    # Calculate a cumulative Net CO2 Flux per year:
    # Sum all samples Net_Long_Term_Budget
    #cumulative_net = df_copy['Net_Long_Term_Budget'].sum()
    #print(f'The cumulative net CO2 emission flux per year is: {cumulative_net} ktCO2/yr')
    
    
    
    
    return df_copy
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

        
    
    
    
    