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
# - Code used to extract Ca/Na and Mg/Na ratios from whole rock analyses of mafic igneous rocks from peru
# - Using data not quite from our area, the function calculates the mols/100g of the elements [as whole rock data is given in %]
# - Then calculates the Ca/Na and Mg/Na ratios
# - As these are log distributed, it then calculates the mean and std dev of these in log space, which is then outputted. 
# But really this should be the mode, and should output the std error




def RockData():
    
    ################## DATA PROCESSING ##################
    
    df_rock = pd.read_excel('chemweathering/data/Mock-Data.xlsx')
    
    df_rock = df_rock[df_rock['Country'] == 'Peru']
    
    df_rock = df_rock[df_rock['Sample_type'].str.contains('metamorphic')==False]  # drop metamorphic
    
    df_rock.dropna(subset=['CaO', 'MgO', 'Na2O'], inplace=True)
    
    ######################################################
    
    
    def compute_molar_concentration(df, element, oxide, molar_masses):
        df[f'{element} (mol/100g)'] = df[oxide] / (molar_masses[element] + molar_masses['O'])
        return df
    
    molar_masses = {
        'Ca': 40.078,
        'Mg': 24.305,
        'Na': 22.990,
        'O': 15.999
    }
    
    ######################################################

    # Define a function to calculate the ratios and log values
    def compute_ratios_and_logs(df):
        df['CaNa'] = df['Ca (mol/100g)'] / df['Na (mol/100g)']
        df['MgNa'] = df['Mg (mol/100g)'] / df['Na (mol/100g)']
        df['LogCaNa'] = np.log(df['CaNa'])
        df['LogMgNa'] = np.log(df['MgNa'])
        return df
    
    
    ######################################################

    # Define a function to compute the mean and standard deviation
    def compute_stats(df):
        mean_log_ca_na = df['LogCaNa'].mean()
        std_log_ca_na = df['LogCaNa'].std()
        mean_log_mg_na = df['LogMgNa'].mean()
        std_log_mg_na = df['LogMgNa'].std()
    
        #print(f'Mean Log Ca/Na is: {mean_log_ca_na}')
        #print(f'Std Dev Log Ca/Na is: {std_log_ca_na}')
        #print(f'Mean Log Mg/Na is: {mean_log_mg_na}')
        #print(f'Std Dev Log Mg/Na is: {std_log_mg_na}')
    
        return mean_log_ca_na, std_log_ca_na, mean_log_mg_na, std_log_mg_na


    ######################################################
    

    # List of elements and their corresponding oxides
    elements_oxides = {
        'Ca': 'CaO',
        'Mg': 'MgO',
        'Na': 'Na2O'
    }

    ######################################################
        

    # Calculate molar concentrations for each element
    for element, oxide in elements_oxides.items():
        if element == 'Na':  # Special case for Na2O to account for two Na atoms
            df_rock[f'{element} (mol/100g)'] = (df_rock[oxide] / (2 * molar_masses[element] + molar_masses['O'])) * 2
        else:
            df_rock = compute_molar_concentration(df_rock, element, oxide, molar_masses)

    # Apply the ratio and log calculations
    df_rock = compute_ratios_and_logs(df_rock)

    # Compute and print the statistics
    mean_log_ca_na, std_log_ca_na, mean_log_mg_na, std_log_mg_na = compute_stats(df_rock)
    
    ######################################################
    
    

    
    
    
    
    ##########################################################################################################
    ################## Maximum Likelihood Estimation for log-normal distribution parameters ##################

    log_mg_na = df_rock['LogMgNa'].replace([np.inf, -np.inf], np.nan).dropna()
    log_ca_na = df_rock['LogCaNa'].replace([np.inf, -np.inf], np.nan).dropna()
    

    mu_mle_mg, sigma_mle_mg = norm.fit(log_mg_na)
    mu_mle_ca, sigma_mle_ca = norm.fit(log_ca_na)
    
    processed_mean_mg_na = math.exp(mu_mle_mg + 0.5*((sigma_mle_mg)**2))
    
    processed_sigma_mg_na = np.sqrt((math.exp(sigma_mle_mg**2) -1)*(math.exp(2*mu_mle_mg + (sigma_mle_mg**2))))
    
    processed_mean_ca_na = math.exp(mu_mle_ca + 0.5*((sigma_mle_ca)**2))
    
    processed_sigma_ca_na = np.sqrt((math.exp(sigma_mle_ca**2) -1)*(math.exp(2*mu_mle_ca + (sigma_mle_ca**2))))
    
    
    
    #print(f'MLE estimated μ for Log Mg/Na: {mu_mle_mg}')
    #print(f'MLE estimated σ for Log Mg/Na: {sigma_mle_mg}')
    #print(f'Estimated μ for Mg/Na: {processed_mean_mg_na}')
    #print(f'Estimated σ for Mg/Na: {processed_sigma_mg_na}')
    
    
        
    #print(f'MLE estimated μ for Log Ca/Na: {mu_mle_ca}')
    #print(f'MLE estimated σ for Log Ca/Na: {sigma_mle_ca}')
    #print(f'Estimated μ for Ca/Na: {processed_mean_ca_na}')
    #print(f'Estimated σ for Ca/Na: {processed_sigma_ca_na}')
    
    
    
    
    
    
    
    ###############################################################################
    ################## PLOT HISTOGRAMS AND FITTED DISTRIBUTIONS ##################
    
    plt.figure(figsize=(10,6))
    plt.hist(df_rock['MgNa'], bins=100, density=True, alpha=0.6, color='g')
    
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    
    #This creates 100 equally spaced values (x) between xmin and xmax. These values are used to generate the points on the fitted log-normal distribution curve.
    
    p = norm.pdf(np.log(x), mu_mle_mg, sigma_mle_mg)
    
    # np.log(x) takes the natural logarithm of each value in x because the data is assumed to follow a log-normal distribution.
    # norm.pdf() computes the probability density function (PDF) of the normal distribution for these log-transformed x values, 
    # using the MLE-estimated parameters mu_mle and sigma_mle.
    
    plt.plot(x, p, 'k', linewidth=2)
    
    #plotting the fitted distribution
    
    plt.title('Mg/Na Histogram and Fitted Log-Normal Distribution')
    plt.xlabel('Mg/Na')
    plt.ylabel('Density')
    plt.savefig('chemweathering/figures/HistogramMgNa.png')
    plt.close()
    #plt.show()
    
    #################################################################################################
    #################################################################################################
    
    
    
    # Plot histogram and fitted distribution
    plt.figure(figsize=(10,6))
    plt.hist(df_rock['CaNa'], bins=100, density=True, alpha=0.6, color='g')
    
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    
    #This creates 100 equally spaced values (x) between xmin and xmax. These values are used to generate the points on the fitted log-normal distribution curve.
    
    p = norm.pdf(np.log(x), mu_mle_ca, sigma_mle_ca)
    
    # np.log(x) takes the natural logarithm of each value in x because the data is assumed to follow a log-normal distribution.
    # norm.pdf() computes the probability density function (PDF) of the normal distribution for these log-transformed x values, 
    # using the MLE-estimated parameters mu_mle and sigma_mle.
    
    plt.plot(x, p, 'k', linewidth=2)
    
    #plotting the fitted distribution
    
    plt.title('Ca/Na Histogram and Fitted Log-Normal Distribution')
    plt.xlabel('Ca/Na')
    plt.ylabel('Density')
    plt.savefig('chemweathering/figures/HistogramCaNa.png')
    plt.close()
    #plt.show()
    
    

    
    
    ##########################################################################################
    
    
    
    
    return(processed_mean_mg_na, processed_sigma_mg_na, processed_mean_ca_na, processed_sigma_ca_na)

