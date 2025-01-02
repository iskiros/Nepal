import sys
import os
import math
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd





def timeseries(df):
    
    #filter df by those which have season column =  'Thalo_timeseries'
    df = df[df['Season'] == 'Thalo_timeseries']
    
    # Make individual graphs of elements over time. Time is given by the "Date" column, which is in the DD/MM/YY format.
    #elements = ['Ca_ppm', 'Fe_ppm', 'K_ppm', 'Li_ppm', 'Mg_ppm', 'Mn_ppm', 'Na_ppm', 'S_ppm', 'Si_ppm', 'Sr_ppm', 'd13C_DIC']
    
    #elements = ['Al/Ca', 'Ba/Ca', 'Fe/Ca', 'K/Ca', 'Li/Ca', 'Mg/Ca', 'Mn/Ca', 'Na/Ca', 'S/Ca', 'Si/Ca', 'Sr/Ca', 'Na/Si', 'Na/Cl', 'Li/Na']
    
    elements = ['Na/Ca']
    
    
    # First, we need to convert the date to a datetime object:
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    
    # Now, we can sort the dataframe by date:
    df = df.sort_values(by='Date')
    
    # Plot Na/Ca against Date
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Na/Ca'], label='Na/Ca')
    plt.xlabel('Date')
    plt.ylabel('Na/Ca')
    plt.title('Na/Ca Time Series')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    
    
    # # Create a PDF to save the plots
    # with PdfPages('thalo_timeseries_ratios_plots.pdf') as pdf:
    #     fig, axs = plt.subplots(len(elements), 1, figsize=(8.27, 11.69))  # A4 size in inches (8.27 x 11.69)
        
    #     for i, element in enumerate(elements):
    #         ax = axs[i]
    #         ax.plot(df['Date'], df[element], label=element)
            
    #         if i == 0:
    #             ax.set_title(f'Thalo Time Series')
    #         if i == len(elements) - 1:
    #             ax.set_xlabel('Date')
    #         if i == len(elements) // 2:
    #             ax.set_ylabel('Concentration')
    #         ax.legend(loc='upper right')
            
    #         ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    #         plt.xticks(rotation=45)
    #         plt.tight_layout()
        
    #     plt.show()
        
    #     # Save the current figure to the PDF
    #     pdf.savefig(fig)
    #     plt.close(fig)
    
    
    
    
def perform_pca(df):
    # Filter the dataframe by 'Thalo_timeseries'
    df = df[df['Season'] == 'Thalo_timeseries']

    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    
    # Sort the dataframe by date
    df = df.sort_values(by='Date')
    
    
    
    # Elements for PCA
    #elements = ['Ca_ppm', 'Fe_ppm', 'K_ppm', 'Li_ppm', 'Mg_ppm', 'Mn_ppm', 'Na_ppm', 'S_ppm', 'Si_ppm', 'Sr_ppm', 'd13C_DIC']

    elements = ['Al/Ca', 'Ba/Ca', 'Fe/Ca', 'K/Ca', 'Li/Ca', 'Mg/Ca', 'Mn/Ca', 'Na/Ca', 'S/Ca', 'Si/Ca', 'Sr/Ca', 'Na/Si', 'Li/Na']
    

    #print(df.shape)

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[elements])

    # Perform PCA
    pca = PCA(n_components=2)  # Let's start with 2 principal components
    pca_result = pca.fit_transform(scaled_data)
    

    # Get the loadings (contributions of each element to each principal component)
    loadings = pd.DataFrame(pca.components_.T, index=elements, columns=['PCA1', 'PCA2'])
    print(loadings)

    # Add the PCA results to the dataframe
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['PCA1'], label='PCA1')
    plt.plot(df['Date'], df['PCA2'], label='PCA2')
    plt.xlabel('Date')
    plt.ylabel('PCA Components')
    plt.title('PCA Time Series Analysis')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
  
    

def plot_superimposed_elements_with_multiple_y_axes(df):
    
    #filter df by those which have season column =  'Thalo_timeseries'
    df = df[df['Season'] == 'Kyul_timeseries']
    
    # Define the elements you want to superimpose
    # Define the elements you want to superimpose
    
    
    #elements = ['Na_ppm', 'Li_ppm', 'K_ppm', 'Ca_ppm', 'Si_ppm', 'Sr_ppm']  # Replace these with PCA1 elements

    #elements = ['Na_ppm', 'S_ppm', 'Fe_ppm', 'Si_ppm', 'Ca_ppm', 'Sr_ppm', 'K_ppm', 'Mg_ppm', 'Li_ppm', 'd13C_DIC'] 
    
    elements = ['Na/Ca', 'Si/Ca', 'Sr/Ca', 'Al/Ca']
    
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    
    # Sort dataframe by date
    df = df.sort_values(by='Date')
    
    # Create a PDF to save the plots
    with PdfPages('superimposed_thalo2_ratios_timeseries_plots.pdf') as pdf:
        fig, ax1 = plt.subplots(figsize=(16, 8))  # A4 size in inches (8.27 x 11.69)
        
        # Set up the first axis (primary y-axis) on the right
        ax1.plot(df['Date'], df[elements[0]], label=elements[0], color='b')
        ax1.set_xlabel('Date')
        ax1.set_ylabel(f'{elements[0]} (ppm)', color='b')
        ax1.yaxis.set_label_position("left")  # Move label to the right
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        ax1.tick_params(axis='y', labelcolor='b')
        #ax1.yaxis.tick_right()  # Move ticks to the right
        
        # Create new axes for each additional element, shifted outwards on the left side
        colors = ['g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']  # Colors for the other elements
        axes = [ax1]  # List to store all axes
        for i, element in enumerate(elements[1:]):
            ax_new = ax1.twinx()  # Create a new y-axis
            ax_new.spines['left'].set_position(('outward', 60 * (i + 1)))  # Shift the axis outward
            ax_new.plot(df['Date'], df[element], label=element, color=colors[i])
            ax_new.set_ylabel(f'{element} (ppm)', color=colors[i])
            ax_new.tick_params(axis='y', labelcolor=colors[i])
            ax_new.yaxis.tick_left()  # Move ticks to the left for all axes
            ax_new.yaxis.set_label_position("left")  # Ensure label is on the left side
            axes.append(ax_new)  # Add the new axis to the list

        # Combine all legends into one
        lines, labels = [], []
        for ax in axes:
            line, label = ax.get_legend_handles_labels()
            lines.extend(line)
            labels.extend(label)
        ax1.legend(lines, labels, loc='upper left')
        
        # Add a title
        ax1.set_title('Thalo, Element Ratios with a strong correlation \n Na/Ca, Si/Ca, Sr/Ca, Al/Ca')
        #with a strong correlation (1) \n Na, Li, K, Ca, Si, Sr

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Show the plot
        plt.show()

        # Save the current figure to the PDF
        pdf.savefig(fig)
        plt.close(fig)   
    
    
    
def pca2(df2):
    
    df = df2.copy()
    
    # Filter the dataframe
    df = df[df['Season'] == 'Thalo_timeseries']

    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    # Sort the dataframe by date
    df = df.sort_values(by='Date')
    
    print(df.columns)
    
    # calculate ratios for elements. concentrations are in ppm so need to convert to mM first
    elements = ['Al_ppm', 'Ba_ppm', 'Ca_ppm', 'Fe_ppm', 'K_ppm', 'Li_ppm', 'Mg_ppm', 'Mn_ppm', 'Na_ppm', 'S_ppm', 'Si_ppm', 'Sr_ppm']
    
    
    # Convert concentrations to mM from ppm which means divide by Atomic weight of the element
    
    # Atomic weights of elements
    weight = {'Al': 26.98,'Ca': 40.08, 'Ba': 137.33, 'Fe': 55.85, 'K': 39.10, 'Li': 6.94, 'Mg': 24.31, 'Mn': 54.94, 'Na': 22.99, 'S': 32.06, 'Si': 28.09, 'Sr': 87.62, 'C': 12.01}
    
    # Convert concentrations to mM
    for element in elements:
        element_name = element.replace('_ppm', '')
        
        df[element_name] = df[element] 
        
        df[element_name + '_mM'] = df[element_name] / weight[element_name]
        
        # remove df[element_name]
        del df[element_name]
        
        
    # Calculate ratios
    
    # Ratios to calculate
    ratios = ['Al/Ca', 'Ba/Ca', 'Fe/Ca', 'K/Ca', 'Li/Ca', 'Mg/Ca', 'Mn/Ca', 'Na/Ca', 'S/Ca', 'Si/Ca', 'Sr/Ca', 'Na/Si', 'Li/Na']
    
    for ratio in ratios:
        elements = ratio.split('/')
        df[ratio] = df[elements[0] + '_mM'] / df[elements[1] + '_mM']    


    # Elements for PCA
    #elements = ['Al/Ca', 'Ba/Ca', 'Fe/Ca', 'K/Ca', 'Li/Ca', 'Mg/Ca', 
                #'Mn/Ca', 'Na/Ca', 'S/Ca', 'Si/Ca', 'Sr/Ca', 'Na/Si', 'Li/Na']
    
    #elements = ['Ca_mM', 'Si_mM', 'Sr_mM', 'Al_mM', 'Na_mM', 'Li_mM', 'K_mM', 'Mg_mM', 'Mn_mM', 'S_mM', 'Ba_mM', 'Fe_mM']


    elements = ['Ca_mM', 'Si_mM', 'Sr_mM', 'Na_mM', 'K_mM', 'Mg_mM', 'S_mM', 'Al_mM']






    # PCA is affected by scale, so you need to scale the features in your data before applying PCA. 
    # Use StandardScaler to help you standardize the data set’s features onto unit scale (mean = 0 and variance = 1), 
    # which is a requirement for the optimal performance of many machine learning algorithms. 
    # If you don’t scale your data, it can have a negative effect on your algorithm

    x = df.loc[:, elements].values
    
    y = df.loc[:,['Date']].values

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(x)

    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    

    # Get the loadings
    loadings = pd.DataFrame(pca.components_.T, index=elements, columns=['PCA1', 'PCA2'])
    print("PCA Loadings:\n", loadings)

    # Add PCA results to dataframe
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]

    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    print("Explained Variance Ratio:", explained_variance)


    # PC1 vs PC2 Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(df['PCA1'], df['PCA2'], c='blue', edgecolor='k', alpha=0.7)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    
    
    # Add arrows for loadings
    for element, (x, y) in loadings.iterrows():
        plt.arrow(0, 0, x, y, color='red', alpha=0.7, head_width=0.05, length_includes_head=True)
        plt.text(x * 1.2, y * 1.2, element, color='red', fontsize=10)

    
    plt.xlabel('Principal Component 1 (PCA1)', fontsize=12)
    plt.ylabel('Principal Component 2 (PCA2)', fontsize=12)
    plt.title('PC1 vs. PC2 Scatter Plot - THALO', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('PCA1_vs_PCA2_Thalo.pdf')
    plt.show()      
        
    
df = pd.read_excel('Datasets/Nepal Master Sheet.xlsx', sheet_name='Final_compiled')





#timeseries(df)

#perform_pca(df)

#plot_superimposed_elements_with_multiple_y_axes(df)

pca2(df)