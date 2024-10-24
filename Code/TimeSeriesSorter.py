import matplotlib.pyplot as plt
import sys
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import os
import pandas as pd
import glob


df = pd.read_excel('/Users/enrico/Desktop/Part III Project/DataSets/Time Series Metadata.xlsx')

print(df.head())

# Create a new column called NEW_SampleID
# This column will look at the locality and the date of the sample and create a unique identifier for each sample
# The format will be to look at locality and date and then add a number to the end of the string
# Eg NTS24_Locality_01 where 01 is the first in the series
# the Date is only used to determine the order in the series, where the oldest is the first in the series
# There are three localities, Thalo, Ke and Bk.
# For Thalo use the shorter TH and for Ke use the shorter KE and for Bk use the shorter BK

# First, we need to map the locality names to their shorter versions
locality_map = {
    'Thalo': 'TH',
    'Ke': 'KE',
    'Bk': 'BK'
}

# Add a new column for the shorter locality names
df['Short_Locality'] = df['Locality'].map(locality_map)

# Sort the dataframe by locality and date
df = df.sort_values(by=['Short_Locality', 'Gregorian Date'])

# Create a new column for the unique sample ID
df['NEW_SampleID'] = df.groupby('Short_Locality').cumcount() + 1

# Format the NEW_SampleID to include the locality and the number
df['NEW_SampleID'] = 'NTS24_' + df['Short_Locality'] + '_' + df['NEW_SampleID'].astype(str).str.zfill(2)

print(df.head())

df.to_excel('/Users/enrico/Desktop/Part III Project/DataSets/Time Series Metadata Altered.xlsx', index=False)