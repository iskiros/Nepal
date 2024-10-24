import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


########################### FUNCTION DEFINITION ###########################

# - What Al sent me at the beginning of the internship to kickstart how I should lay out my python, as well as my plot
# - Contains many functions and has since been unused and turned into other functions






def slice_relevant_data(df):
    """Slice relevant data from the dataset"""
    # select major elements for charge balance
    elements = {'Ca', 'Mg', 'K', 'Na', 'HCO3', 'Cl', 'SO4', 'Al', 'As', 'Ba', 'Bi', 'Ce', 'Fe', 'Li', 'Rb', 'Sr', 'CO3', 'H2PO4'}

    # loop through columns to retrieve columns with major elements in name with element+_in_mg_kg-1
    # save as a list
    major_elements = []
    for element in elements:
        col_standard = f"{element}_in_mg_kg-1"
        col_alternate = f"{element}_in_mg_kg-1 [MA-0722]"
        if col_standard in df.columns:
            major_elements.append(col_standard)
        elif col_alternate in df.columns:
            major_elements.append(col_alternate)

    # slice the dataframe to include only the major elements
    df_slice = df[major_elements]

    # append the sample ID to the sliced dataframe
    df_slice.insert(0, 'unique_code', df['unique_code'])

    return df_slice

def calculate_molar_concentration(df):
    """Calculate molar concentration of the major elements"""
    element_dict = {
        'molecules': ['Ca', 'Mg', 'K', 'Na', 'HCO3', 'Cl', 'SO4', 'Al', 'As', 'Ba', 'Bi', 'Ce', 'Fe', 'Li', 'Rb', 'Sr', 'CO3', 'H2PO4'],
        'molar_mass': [40.08, 24.31, 39.10, 22.99, 61.02, 35.45, 96.06, 26.98, 74.92, 137.33, 208.98, 140.12, 55.85, 6.94, 85.47, 87.62, 60.01, 97.99],
        'valency': [2, 2, 1, 1, 1, 1, 2, 3, 3, 2, 3, 3, 2, 1, 1, 2, 2, 1]
    }

    # Suffixes
    suffix = '_in_mg_kg-1'
    alt_suffix = '_in_mg_kg-1 [MA-0722]'

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
    df_copy = df_copy[['unique_code'] + [col for col in df_copy.columns if ' [aq] (mM)' in col] + [col for col in df_copy.columns if ' [aq] (meq/L)' in col]]

    #print(df_copy)

    return df_copy


def charge_balance(df):
    """Function to charge balance"""
    cations = ['Ca', 'Mg', 'K', 'Na', 'Al', 'As', 'Ba', 'Bi', 'Fe', 'Li', 'Rb', 'Sr']
    anions = ['HCO3', 'Cl', 'SO4', 'H2PO4', 'CO3']

    # Calculate the sum of cations and anions
    df['sum_cations'] = df[[col for col in df.columns if any(cation in col for cation in cations) and ' [aq] (meq/L)' in col]].sum(axis=1)

    df['sum_anions'] = df[[col for col in df.columns if any(anion in col for anion in anions) and ' [aq] (meq/L)' in col]].sum(axis=1)

    # NICB calculation
    df['NICB diff'] = (df['sum_cations'] - df['sum_anions'])
    df['NICB sum'] = (df['sum_cations'] + df['sum_anions'])

    # Remove rows with NICB sum = 0
    df = df[df['NICB sum'] != 0]

    df['NICB'] = df['NICB diff'] / df['NICB sum']

    return df

def plot_NICB(df):
    """Plot NICB distribution"""
    # Plot the NICB distribution as a histogram
    fig, ax = plt.subplots()

    # Plot KDE
    df['NICB'].plot.kde(ax=ax, legend=False, title='NICB distribution')

    # Save plot
    plt.savefig('chemweathering/figures/NICB_distribution2.png')

def GDPlot(df):

    #print(df.columns)

    plt.figure(figsize=(10,6))
    plt.scatter(df['Ca [aq] (mM)'], df['HCO3 [aq] (mM)'], color='blue', alpha=0.7, s=70)
    plt.xlabel('Ca [aq] (mM)')
    plt.ylabel('HCO3 [aq] (mM)')
    plt.title('Gaillardet Plot')
    plt.savefig('Gaillardet_plot.pdf')
    plt.show()

def NICB():
    """The main NICB function to calculate NICB values for a given dataset"""
    # Import data
    df = pd.read_excel('chemweathering/data/Canete_Long_Data_Revisited_Local.xlsx', sheet_name='Data')
    
    # Slice waters
    df = df[df['sample_type'] == 'water']

    # Slice relevant data
    df_slice = slice_relevant_data(df)

    # Replace non-numeric values with NaN in the dataframe except the first column
    df_slice.loc[:, df_slice.columns[1:]] = df_slice.loc[:, df_slice.columns[1:]].apply(pd.to_numeric, errors='coerce')

    # Create molar columns
    df_neat = calculate_molar_concentration(df_slice)

    # Charge balance
    df_neat = charge_balance(df_neat)

    # Generate Gaillardet plot
    GDPlot(df_neat)

if __name__ == '__main__':
    NICB()







