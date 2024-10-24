#!/usr/bin/env python3
"""Main module."""

# Import Python Modules
import os
from metadata import metadata
from standards import extstdcorr
import pandas as pd # pandas
from pandas import ExcelWriter
import sys

def userinoutfiles():
    """ User defined input (CSV) and output (xlsx) paths and files"""

    # Input path and file
    print("Name of input file?")
    ipf = input("")
    print("Input file:", ipf)

    # Output path and file
    print("Name of output file?")
    default_output_file = os.path.splitext(ipf)[0]+'_out.xlsx'
    print("Default output file:", default_output_file)
    opf = input("") # or default_output_file
    if opf == "":
        opf = default_output_file
    print("Output file:", opf)
    return(ipf, opf)


def usermetadata(ipf):
    """ Optional user metadata input """

    md_dsn = input("Input metadata? (Y/N): ")
    if md_dsn == 'Y':
        agilent_df = pd.read_csv(ipf, header=None)
        mddf = metadata(agilent_df)
    else:
        nmd = {'': ['No user metadata input']}
        no_metadata = pd.DataFrame(data=nmd)
        mddf = no_metadata
    return mddf, md_dsn


def usermetadata(ipf):
    """ Optional user metadata input """
    agilent_df = pd.read_csv(ipf, header=None)
    mddf = metadata(agilent_df)
    return mddf


def agilentcsvdataimport(ipf):
    """ Import Agilent Data from csv and convert to useable
    data form"""

    # Specify data type for agilent csv columns
    col_dtype = {'Concentration': float, 'Intensity': float,
                 'Concentration SD': float, 'Concentration % RSD': float,
                 'Intensity SD': float, 'Intensity % RSD': float,
                 'Weight': float, 'Volume': float, 'Dilution': float,
                 'Correlation coefficient': float,
                 'Calibration Coefficient [1]': float,
                 'Calibration Coefficient [2]': float,
                 'Calibration Coefficient [3]': float}

    # Import Agilent csv data
    aglnt_df = pd.read_csv(ipf, skiprows=5, dtype=col_dtype,
                       na_values=['--', '####', 'Uncal', '-', '> 100.00'])
    return aglnt_df


def agilentdf_format(aglnt_df):
    """ Format Raw Agilent Data"""

    # Date time to format: DD/MM/YYYY HH:MM:SS
    aglnt_df['Date Time']= pd.to_datetime(aglnt_df['Date Time'],
                                      format='%d/%m/%Y %H:%M:%S')

    # Column manipulation, insertion, and deletion
    aglnt_df.rename(columns={'Element':'Element & Wavelength (nm)'},
                    inplace=True)

    # Insert columns
    loc = aglnt_df.columns.get_loc("Element & Wavelength (nm)")

    def col_split(df, loc, name, col, word):
        df.insert(loc, name, df[col].str.split(' ', n=1, expand=True)[word])

    col_split(aglnt_df, loc, 'View', 'Element Label', 1)
    col_split(aglnt_df, loc, 'Wavelength (nm)',
                   'Element & Wavelength (nm)', 1)
    col_split(aglnt_df, loc, 'Element', 'Element & Wavelength (nm)', 0)

    aglnt_df['Column Label'] = aglnt_df[['Element','View',
                                 'Wavelength (nm)','Unit']].agg(' '.join,
                                                                axis=1)
    aglnt_df["Wavelength (nm)"] = pd.to_numeric(aglnt_df["Wavelength (nm)"])

    # Delete unecessary columns
    aglnt_df = aglnt_df.drop(['Element & Wavelength (nm)'], axis=1)
    aglnt_df = aglnt_df.drop(['Element Label'], axis=1)

    return aglnt_df


def flagremoval(aglnt_df):
    """ Remove data with over-range and under-range flags"""
    aglnt_df = (aglnt_df[(aglnt_df['Flags'] != 'o')
                   & (aglnt_df['Flags'] != 'u')])
    return aglnt_df


def processingsteps(processes):
    """ Produce dataframe of processing steps performed"""

    processing_data = {'Process':  ['Metadata input?',
                                    'Flag removal?',
                                    'Wavelengths Removed:'],
                       'Action': processes}
    processing_df = pd.DataFrame(processing_data)
    return processing_df

def pddftoxlsx(dict_df, path):
    """ Write pandas dataframes to sheets in xlsx file"""
    writer = ExcelWriter(path, engine='openpyxl')  # Specify the engine as 'openpyxl'
    for key in dict_df.keys():
        dict_df[key].to_excel(writer, sheet_name=key, index=False, na_rep='NaN')

    writer.close()

def main():
    """ Operations in agilentexporttools"""
    # User-defined input and output files
    ipf, opf = userinoutfiles()

    # Optional user metadata input
    md_dsn = input("Input metadata? (Y/N): ")
    if md_dsn == 'Y':
        md_df = usermetadata(ipf)
    else:
        nmd = {'': ['No user metadata input']}
        no_metadata = pd.DataFrame(data=nmd)
        md_df = no_metadata
        md_dsn = 'N'

    aglt_df = agilentcsvdataimport(ipf)
    aglt_raw = agilentdf_format(aglt_df)

    # Optional removal of flagged data
    flags_dsn = input("Remove data with flags? (Y/N): ")
    if flags_dsn == 'Y':
        aglnt_df = flagremoval(aglt_raw)
    else:
        flags_dsn == 'N'

    # Wavelength Selector
    pconc, estd, wl_rm = extstdcorr(aglt_df)

    # Decision log
    decisions = [md_dsn, flags_dsn, wl_rm]
    process_df = processingsteps(decisions)

    # Excel sheet output
    xlsxdict = {
        "Metadata": md_df,
        "Processing Log": process_df,
        "Raw": aglt_raw,
        "Standard Stats": estd,
        "Conconcentration (Processed)": pconc}
    pddftoxlsx(xlsxdict, str(opf))

if __name__ == "__main__":
    main()
