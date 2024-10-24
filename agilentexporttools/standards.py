#!/usr/bin/env python3
import os
import pandas as pd
import sys
import numpy as np

def estdimport(filepath="data/standards.csv"):
    """ import external standard database"""
    script_dir = os.path.dirname(__file__)
    estd_path = os.path.join(script_dir, filepath)
    std_df = pd.read_csv(estd_path)
    return std_df


def estdprint(agilent_df, estd_df):
    """ Print standards in database and standards used during Agilent run"""
    sqc = agilent_df[agilent_df['Type'] == 'QC']
    estd_ls = sqc['Label'].unique()
    print('External Standards:', estd_ls)
    print("External standards in database: ",
          estd_df['External Standard Label'].unique())
    return estd_ls


def estdmatch(estd_ls):
    """ Match standards used to standards in database"""
    usn = []

    for estd in estd_ls:
        print('Name of Standard: ', estd)
        stdname = input("Name of Standard in Database: ")
        dilf = input("Dilution Factor: ")
        usn.append([estd, stdname, float(dilf)])

    usn_cols = ['Label', 'External Standard Label', 'Dilution Factor']
    usn_df = pd.DataFrame(usn, columns=usn_cols)
    return usn_df


def estdmerge(data, estd_df, estd_ls):
    """ Merge standard database and Agilent Dataframe"""

    # Add standard concentration labels to agilent dataframe
    aglt_df = pd.merge(data, estd_ls, on='Label')

    # Adjust standard concentrations to literature concentrations
    aglt_df['UD Concentration'] = (aglt_df['Concentration']*
                                   aglt_df['Dilution Factor'])

    # Add standard concentration data to agilent dataframe
    mcs = ['External Standard Label', 'Element', 'Standard Concentration',
           'Standard 2SD', 'Standard Unit']
    std_comp = pd.merge(aglt_df, estd_df[mcs],
                        how = 'inner',
                        on = ['Element', 'External Standard Label'])
    return std_comp


def estdaccuracy(obvs, exp):
    """Compare the accuracy of external standard measurements to literature
    values"""
    # Absolute deviation
    dev = abs(((obvs-exp)/exp))

    # Percentage deviation
    pcf = 1e2
    dstd = dev*pcf
    return dstd

def estdpivot(data, vls):
    """ Pivot agilent dataframe"""
    data_pvt = pd.pivot_table(data, index=['Date Time', 'Label'],
                              columns=['Column Label'], values=vls,
                              aggfunc='first', sort=False)
    data_pvt.reset_index(inplace=True)
    return data_pvt

def extstdcorr(data):
    """Filter Agilent wavelengths based on external standards"""

    # Import external standards
    ext_standards = estdimport()

    # Print list of standards in Agilent data and database
    match_ls = estdprint(data, ext_standards)

    # Match database and Agilent standards
    std_lb = estdmatch(match_ls)

    # Merge database and Agilent dataframes
    std_c = estdmerge(data, ext_standards, std_lb)

    # Calculate deviation from literature value
    obs = std_c['UD Concentration']
    expt = std_c['Standard Concentration']
    std_c['Deviation from standard (%)'] = estdaccuracy(obs, expt)

    # Pivot data
    standard_pd = estdpivot(std_c, 'Deviation from standard (%)')
    standard_pd_sts = estdpivot(std_c, 'UD Concentration')
    
    # Select all wavelengths
    lambdas = list(standard_pd.columns[3:])

    u_std = standard_pd.groupby(['Label'])[lambdas].mean().reset_index()
    stdev = standard_pd_sts.groupby(['Label'])[lambdas].std().reset_index()
    mean = standard_pd_sts.groupby(['Label'])[lambdas].mean().reset_index()
    rstd = stdev.loc[:, lambdas].div(mean.loc[:, lambdas])*1e2

    u_std = u_std.drop('Label', axis=1)

    new_df = u_std.T.rename(columns=std_lb['Label'])
    new_df = new_df.add_suffix(' Dev. (%)')

    rstd = rstd.T.rename(columns=std_lb['Label'])
    rstd = rstd.add_suffix(' RSD (%)')

    CMBD = pd.merge(new_df, rstd, left_on=['Column Label'],
                    right_on=['Column Label'])
    CMBD.reset_index(inplace=True)

    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None):
        print(CMBD)

    # Remove bad wavelengths
    input_string = input("Wavelengths to Remove: ")
    user_list = input_string.split(", ")
    processed_data = data[~data['Column Label'].isin(user_list)]

    std_c = std_c[std_c['Column Label'].isin(user_list)]
    std_c['Deviation from standard (%) - mean'] = std_c.groupby(['Label', 'Element'])['Deviation from standard (%)'].transform('mean')

    selected_data = processed_data[processed_data['Type']=='Sample']

    mean_conc = selected_data.groupby(['Label', 'Element', 'Date Time'])['Concentration'].transform('mean')

    selected_data.insert(3, 'Mean conc (ppm)', mean_conc)

    processed_concs = pd.pivot_table(selected_data, index=['Label'],
                                     columns=['Element'],
                                     values='Mean conc (ppm)',
                                     aggfunc='mean', sort=False)
    processed_concs.reset_index(inplace=True)
    keep_same = {'Label'}
    processed_concs.columns = ['{}{}'.format(c, '' if c in keep_same else ' (ppm)')
                               for c in processed_concs.columns]

    
    return(processed_concs, CMBD, user_list)
