#!/usr/bin/env python3
"""Metadata module."""

# Import Python Modules
import pandas as pd

def metadata(agilent_dataframe):
    """Metadata for Agilent output file"""

    # Slice metadata
    meta_df = agilent_dataframe.iloc[0:4, 0:2]
    esws_fp = meta_df.iloc[0, 0]
    username = input("Username: ")
    doa = input("Date of Analysis (DD/MM/YYYY): ")
    sample_description = input("Samples (e.g. 'Dissolved Riverine Cations"
                                 " from the Humber'): ")
    matrix = input("Background Matrix (e.g. '0.04M AmCl) : ")
    calibration_line = input("Calibration Line (e.g. Synthetic; 0.04M"
                                 " AmCl'): ")
    nebuliser = input("Nebuliser (e.g. 'Non-carb glass neb"
                                 " 129484'): ")
    comments = input("Comments: ")

    metadata_dict = {'Variable': ["Worksheet export file path",
                              meta_df.iloc[1,0]+': ',
                              meta_df.iloc[2,0]+': ',
                              meta_df.iloc[3,0]+': ',
                              "User: ",
                              "Date of analysis: ",
                              "Samples: ",
                              "Matrix: ",
                              "Calibration line: ",
                              "Nebuliser: ",
                              "Comments: "],
                     'Value': [esws_fp[25:], meta_df.iloc[1,1],
                               meta_df.iloc[2,1], meta_df.iloc[3,1],
                               username, doa, sample_description, matrix,
                               calibration_line, nebuliser, comments]}

    metadata_dataframe = pd.DataFrame(data=metadata_dict)
    return metadata_dataframe
