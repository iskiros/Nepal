
########################### IMPORTING LIBRARIES ###########################

import pandas as pd
from peru_functions_gio.slice_relevant_data import slice_relevant_data
from peru_functions_gio.calculate_molar_concentration import calculate_molar_concentration
from peru_functions_gio.molar_conc_seawater import molar_conc_seawater
from peru_functions_gio.charge_balance import charge_balance
from peru_functions_gio.plot_NICB import plot_NICB
from peru_functions_gio.convert_latitude import convert_latitude
from peru_functions_gio.convert_longitude import convert_longitude
from peru_functions_gio.GDPlot import GDPlot
from peru_functions_gio.haversine_distance import haversine_distance
from peru_functions_gio.DistPlot import DistPlot
from peru_functions_gio.mixdiag import mixdiag
#from functions_gio.oldmap import oldmap
from peru_functions_gio.map_with_log_colorscale import map_with_log_colorscale
from peru_functions_gio.NICB_Valid import NICB_Valid
from peru_functions_gio.Rain_Uncorrected import uncorrected_plots
from peru_functions_gio.Chloride_Correction import Chloride_Correction
from peru_functions_gio.Chloride_Correction2 import Chloride_Correction2
from peru_functions_gio.Chloride_Correction_3 import Chloride_Correction_3
from peru_functions_gio.Chloride_Correction4 import Chloride_Correction4
from peru_functions_gio.Piper import Piper
from peru_functions_gio.RockData import RockData
from peru_functions_gio.Xsil import Xsil
from peru_functions_gio.Xsil_uncorrected import Xsil_uncorrected
from peru_functions_gio.Xsil_NEW import Xsil_NEW
from peru_functions_gio.Bathymetry_Old import Bathymetry_Old
from peru_functions_gio.rotate_points_180 import rotate_points_180
from peru_functions_gio.Bathymetry import Bathymetry
from peru_functions_gio.Bathymetry_Dual import Bathymetry_Dual
from peru_functions_gio.Bathymetry_Heatmap import Bathymetry_Heatmap
from peru_functions_gio.Correction_Sil import Correction_Sil
from peru_functions_gio.sed_analysis import sed_analysis
from peru_functions_gio.carbonate_endmember import carbonate_endmember
from peru_functions_gio.Blattman_Correction import Blattman_Correction
from peru_functions_gio.Activity_Analysis import activityplot
from peru_functions_gio.Fraction_Calculation import fraction_calculation
from peru_functions_gio.manage_areas import manage_areas
from peru_functions_gio.flux_calc import flux_calc




########################### MASTER SCRIPT ###########################

def main():
    
    ####### INITIAL DATAFRAME MANIPULATION #######
    
    # Import data
    df = pd.read_excel('chemweathering/data/Canete_Long_Data_Revisited_Local.xlsx', sheet_name='Data')
    
    # Convert 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Extract month and year from 'date' column
    df['month_year'] = df['date'].dt.strftime('%m%y')
    
    # For samples after row 290, change 'unique_code' to a concatenation of 'site' and 'month_year' because they have the wrong code format
    df.loc[290:, 'unique_code'] = df.loc[290:, 'site'] + '-' + df.loc[290:, 'month_year']
    
    # Filter waters
    df = df[df['sample_type'] == 'water']
    
    ## Remove any rows with the string "UF" in the unique_code column
    df = df[~df['unique_code'].str.contains('UF', na=False)]
    
    #df.to_csv('chemweathering/data/workinprogress.csv', index=False)
    
    ##################################################################
    
    
    
    

    ####### DATA SLICING FOR RELEVANT DATA #######

    # Slice relevant data
    df_slice = slice_relevant_data(df)

    # Replace non-numeric values with NaN in the dataframe except the first column and the last columns, which are unique_code and water_body
    df_slice.loc[:, df_slice.columns[1:-1]] = df_slice.loc[:, df_slice.columns[1:-1]].apply(pd.to_numeric, errors='coerce')
    
    # Drop rows with NaN values in 'unique_code' column
    #df_slice = df_slice.dropna(subset=['unique_code'])
    
    
    ##################################################################




    ####### CALCULATING MOLAR VALUES AND CHARGE BALANCE #######

    # Create molar columns
    df_neat = calculate_molar_concentration(df_slice)
    
    # Charge balance
    df_neat = charge_balance(df_neat)

    # Initialize lists to store unique codes and NICB values 
    NICB_Balance = NICB_Valid(df_neat)
    
    #print(df_neat.columns)
    
    ##################################################################



    
    ####### INITIAL PLOTTING AND ANALYSIS #######
    
    ##################################################################
    
    df_neat.loc[:, 'latitude_converted'] = df_neat.apply(convert_latitude, axis=1)
    df_neat.loc[:, 'longitude_converted'] = df_neat.apply(convert_longitude, axis=1)

    # Drop rows with NaN values in converted latitude and longitude
    df_neat.dropna(subset=['longitude_converted', 'latitude_converted'], inplace=True)
    
    ##################################################################

    
    # Generate Gaillardet plot
    #GDPlot(df_neat, NICB_Balance)
    
    #GD Plot converts the latitudes as well as the longitudes so must be included... to make more streamlined in future
    
    #map_with_log_colorscale(df_neat)
    
    #Map also converts the latitudes as well as the longitudes so must be included
    #Ideally would make this more streamlined. Why does it edit the df directly
    # Has been fixed in the new version

    ##################################################################
    
    
    
    
    
    
    ####### UNCORRECTED DATA INVESTIGATION #######


    ## Function to see uncorrected data
    
    #uncorrected_plots(df_neat, NICB_Balance)

    # plot_NICB(df_neat)
    
    ##################################################################
    
    
    
    
    
    
    
    
    ####### OUR SEDIMENT DATA INVESTIGATION #######
    
    df_sed = sed_analysis()
    
    df_sed.to_excel('chemweathering/data/Canete_Our_Seds.xlsx')
    
    ##################################################################
    
    
    
    
    
    
    ####### PIPER PLOT INVESTIGATION #######
    
    #df_piper = Piper(df_neat)
    
    ##################################################################
    




    ####### CHLORIDE CORRECTIONS #######
    
    ##### df_corrected, df_mock = Chloride_Correction2(df_neat, NICB_Balance) ## TV2 correction
    
    df_mock = Chloride_Correction4(df_neat, NICB_Balance) ## Spatial Rain correction
    
    #df_corrected, df_mock = Blattman_Correction(df_neat, NICB_Balance) ## TV2 correction
    
    #df_corrected = Chloride_Correction(df_neat, NICB_Balance)
    
    #df_corrected = Chloride_Correction_3(df_neat, NICB_Balance) ## Just testing out the new function
    
    #### df_2407 = Correction_Sil(df_neat, NICB_Balance) ### for the simulated X_Sil
    
    ####### TO DATE CC2 is the best #######
    
    ##################################################################
    
    
    # bring df_mock to a csv file:
    #df_mock.to_csv('chemweathering/data/df_mock_for_SIC.csv', index=False)
    
    
    ####### CORRECTED RATIO ANALYSES #######
    
    carbonate_endmember(df_mock)
    
    #mixdiag(df_corrected)
    
    #DistPlot(df_corrected)
    
    ##################################################################
    
    

    
    
    
    
    
    ####### OTHER SEDIMENT DATA ANALYSIS #######
    
    #processed_mean_mg_na, processed_sigma_mg_na, processed_mean_ca_na, processed_sigma_ca_na = RockData()
    
    ##################################################################
    
    
    
    
    
    
    ####### XSIL ANALYSES #######
    
    #df_XSil = Xsil(df_corrected, processed_mean_mg_na, processed_sigma_mg_na, processed_mean_ca_na, processed_sigma_ca_na) What the input should be
    
    
    df_XSil = Xsil(df_mock, 0.36, 0.1, 0.59, 0.1) #Just for the moment
    
    
    #df_XSil_uncorrected = Xsil_uncorrected(df_neat, NICB_Balance, df_sed, 0.36, 0.1, 0.59, 0.1, 0.0565, 0.1) #Just for the moment
    
    
    #df_XSil_NEW = Xsil_NEW(df_corrected, mg_na_sil_ratio=0.5, ca_na_sil_ratio=0.5)
    
    ####### THE "NOT NEW" ONE WORKS WELL FOR THE HALITE CORRECTION #######
    
    ##################################################################
    
    
    
    ####### FRACTION AND AREA CALCULATION #######
    
    df_fraction = fraction_calculation(df_XSil)
    
    
    df_fraction.to_excel('chemweathering/data/fraction_data.xlsx', index=False)
    
    
    ##### ##############################
    # R CODE FOR FRACTION CALCULATION TO CREATE THE AREA DF SPREADSHEET
    ##### ##############################
    
    
    
    
    df_area = manage_areas(df_fraction)
    
    df_flux = flux_calc(df_area)
    
    df_flux.to_excel('chemweathering/data/flux_data.xlsx', index=False)
    
    
    ##################################################################
    
    
    ####### MAP MAKING #######
    
    #Bathymetry(df_flux)
    
    #Bathymetry_Heatmap(df_flux)
    
    Bathymetry_Dual(df_flux)
    
    #lithologydf = Bathymetry(df_XSil)
    
    ##################################################################
    
    #activityplot(lithologydf)


########################### CALLING MAIN FUNCTION ###########################

if __name__ == '__main__':
    main()
