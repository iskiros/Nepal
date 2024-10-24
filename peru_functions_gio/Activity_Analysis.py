import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib import cm

def activityplot(df_mock):
    
    #############################################################################################################
    
    ######### LOAD AND PROCESS DATA #########
    
    # Load the data from Excel
    df = pd.read_excel('chemweathering/data/df_mock_for_SIC.xlsx')

    # Create a copy of the data to avoid modifying the original
    df_copy = df.copy()
    
    ##############################################################################################################
    
    
    
    ######### LITHOLOGY LABELS #########

    # Map lithology labels to the df_copy DataFrame
    df_copy['lithology'] = df_copy['unique_code'].map(df_mock.set_index('unique_code')['lithology_label'])

    # Define colors for each lithology type
    lithology_colours = {'sed': 'blue', 'plutonic': 'purple', 'volcanic': 'red'}
    
    ##############################################################################################################
    
    ######### REMOVE RC14WF BECAUSE IT PLOTS WAY BELOW EVERYTHING ELSE #########

    # Prune specific entries
    df_copy = df_copy[~df_copy['unique_code'].str.contains('RC14WF-1121', na=False)]
    
    ##############################################################################################################



    ######### PLOT TO DETERMINE SATURATION #########


    # Plot the scatter points with different markers for the lithology label
    fig, ax = plt.subplots(figsize=(10, 6))
    for lithology_type, colour in lithology_colours.items():
        subset = df_copy[df_copy['lithology'] == lithology_type]
        ax.scatter(
            subset['Ca+2 Log Activity'],
            subset['CO3-2 Log Activity'],
            c=colour,
            edgecolors='black',
            s=50,
            label=lithology_type,
        )
        
    # Sort by unique_code to maintain the desired order
    df_copy.sort_values(by='unique_code', inplace=True)
    
    
    ######### PLOT LINE BETWEEN POINTS FOR THE RIVERS #########

    # Plot main series line (e.g., RC00 to RC16) with viridis colormap
    main_series = df_copy[df_copy['unique_code'].str.startswith('RC')].copy()
    main_series['numeric_code'] = main_series['unique_code'].str.extract(r'(\d+)').astype(int)

    # Sort main_series by numeric_code
    main_series.sort_values(by='numeric_code', inplace=True)

    # Plot each segment with a color from the colormap
    viridis = cm.get_cmap('viridis', len(main_series))
    for i in range(len(main_series) - 1):
        ax.plot(
            [main_series.iloc[i]['Ca+2 Log Activity'], main_series.iloc[i+1]['Ca+2 Log Activity']],
            [main_series.iloc[i]['CO3-2 Log Activity'], main_series.iloc[i+1]['CO3-2 Log Activity']],
            color=viridis(i / (len(main_series) - 1)),  # Assign color from colormap
            linewidth=2
        )
        
    ##############################################################################################################
    
    
    ######### PLOT LINE BETWEEN POINTS FOR THE TRIBUTARIES #########
    
    
    ##### A BIT OF TLC REQUIRED BECAUSE SOME TRIBUTARIES DON'T HAVE A POINT TO CONNECT TO #####

    # Plot tributary lines (e.g., T1, T2a, T2b)
    tributaries = df_copy[df_copy['unique_code'].str.startswith('T')].copy()
    tributaries.sort_values(by='unique_code', inplace=True)

    # Extract the numeric parts from the tributary codes
    tributary_groups = tributaries['unique_code'].str.extract(r'([A-Za-z]+)(\d+)')
    tributaries['group'] = tributary_groups[0] + tributary_groups[1]
    tributaries['numeric_code'] = tributary_groups[1].astype(int)

    # Process each tributary group
    for name, group in tributaries.groupby('group'):
        numeric_code = int(re.search(r'\d+', name).group())

        # Find the RC point with the same numeric code
        exact_rc = main_series[main_series['numeric_code'] == numeric_code]

        if not exact_rc.empty:
            # Connect tributary to the exact RC point if it exists
            for _, tributary_point in group.iterrows():
                ax.plot(
                    [exact_rc['Ca+2 Log Activity'].values[0], tributary_point['Ca+2 Log Activity']],
                    [exact_rc['CO3-2 Log Activity'].values[0], tributary_point['CO3-2 Log Activity']],
                    linestyle='--', color='gray'
                )
        else:
            # If no exact RC point, use average position between RC0X-1 and RC0X+1
            lower_rc = main_series[main_series['numeric_code'] < numeric_code].sort_values(by='numeric_code').iloc[-1]
            upper_rc = main_series[main_series['numeric_code'] > numeric_code].sort_values(by='numeric_code').iloc[0]

            avg_ca_log = (lower_rc['Ca+2 Log Activity'] + upper_rc['Ca+2 Log Activity']) / 2
            avg_co3_log = (lower_rc['CO3-2 Log Activity'] + upper_rc['CO3-2 Log Activity']) / 2

            # Plot lines connecting the average position to each point in the tributary group
            for _, tributary_point in group.iterrows():
                ax.plot(
                    [avg_ca_log, tributary_point['Ca+2 Log Activity']],
                    [avg_co3_log, tributary_point['CO3-2 Log Activity']],
                    linestyle='--', color='gray'
                )
        
        # Connect the tributary group with a dashed line
        ax.plot(
            group['Ca+2 Log Activity'],
            group['CO3-2 Log Activity'],
            linestyle='--', color='gray', label=f'Tributary {name}'
        )
        
        
        
    ##############################################################################################################
    
    

    # Add labels and annotations
    for index, row in df.iterrows():
        ax.text(row['Ca+2 Log Activity'], row['CO3-2 Log Activity'], row['unique_code'][:4], fontsize=8)

    # Plot the line with equation y = -x - 8.5
    x = np.linspace(-3.75, -2.5, 100)
    y = -x - 8.5
    ax.plot(x, y, c='blue')
    
    ##############################################################################################################
    

    # Make the bit above the line shaded gray
    ax.fill_between(x, y, -3, color='gray', alpha=0.5, zorder=-1)
    ax.text(-3, -3.75, 'Supersaturated', fontsize=12, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.text(-3.5, -5.5, 'Undersaturated', fontsize=12, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))


    ##############################################################################################################

    # Customize plot
    ax.set_title('CO3-2 Log Activity vs Ca+2 Log Activity')
    ax.set_xlabel('Ca+2 Log Activity')
    ax.set_ylabel('CO3-2 Log Activity')
    ax.set_xlim(-3.75, -2.5)
    ax.set_ylim(-5.75, -3.5)
    ax.grid()

    # Add a colorbar to represent RC00 to RC16
    sm = plt.cm.ScalarMappable(cmap=viridis, norm=plt.Normalize(vmin=0, vmax=16))
    sm.set_array(main_series['numeric_code'])  # Set the array with the numeric codes
    fig.colorbar(sm, ax=ax, label='Main Series Index (RC00 - RC16)')

    ax.legend()
    #plt.show()
    plt.close()






    ##############################################################################################################
    ##############################################################################################################




    ######### PLOT TO SHOW CALCITE SATURATION #########

    # Plot Calcite SI** on the y agaist Ca/Na:
    
    df_copy['Ca/Na'] = df_copy['+Ca [aq] (mM)'] / df_copy['+Na [aq] (mM)']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    
    
    # Define colors for each lithology type
    lithology_colours = {'sed': 'o', 'plutonic': '^', 'volcanic': '*'}
    
    
    for lithology_type, markers in lithology_colours.items():
        subset = df_copy[df_copy['lithology'] == lithology_type]
        ax.scatter(
            subset['Ca/Na'],
            subset['Calcite SI**'],
            c='black',
            marker = markers,
            edgecolors='black',
            s=50,
            label=lithology_type,
        )
        
    xlim = df_copy['Ca/Na'].max()    
    
    ##############################################################################################################
    
    
    
    ######### PLOT LINE BETWEEN POINTS FOR THE RIVERS #########

    # Plot main series line (e.g., RC02 to RC16) with viridis colormap
    main_series = df_copy[df_copy['unique_code'].str.startswith('RC')].copy()
    main_series['numeric_code'] = main_series['unique_code'].str.extract(r'(\d+)').astype(int)

    # Filter to start from RC02 onward
    main_series = main_series[main_series['numeric_code'] >= 2]

    # Sort main_series by numeric_code
    main_series.sort_values(by='numeric_code', inplace=True)

    # Plot each segment with a color from the colormap
    viridis = cm.get_cmap('viridis', len(main_series))
    for i in range(len(main_series) - 1):
        ax.plot(
            [main_series.iloc[i]['Ca/Na'], main_series.iloc[i+1]['Ca/Na']],
            [main_series.iloc[i]['Calcite SI**'], main_series.iloc[i+1]['Calcite SI**']],
            color=viridis(i / (len(main_series) - 1)),  # Assign color from colormap
            linewidth=2
        )
    
    ##############################################################################################################

    
        
    # Line at y = 0:
    x = np.linspace(0, xlim, 100)
    y = np.zeros(100)
    ax.plot(x, y, c='black')    
        
    # Add labels and annotations
    for index, row in df_copy.iterrows():
        ax.text(row['Ca/Na'], row['Calcite SI**'], row['unique_code'][:4], fontsize=8)
    
    
    ax.set_xscale('log', base=2)
        
    # Customize plot
    ax.set_title('SI_calcite vs Ca/Na')
    ax.set_xlabel('Ca/Na')
    ax.set_ylabel('SI_calcite')
    ax.set_xlim(0, xlim)
    #ax.set_ylim(-5.75, -3.5)
    ax.grid()
    
    # Add a colorbar to represent RC02 to RC16
    sm = plt.cm.ScalarMappable(cmap=viridis, norm=plt.Normalize(vmin=2, vmax=16))
    sm.set_array(main_series['numeric_code'])  # Set the array with the numeric codes
    fig.colorbar(sm, ax=ax, label='Main Series Index (RC02 - RC16)')

    

    ax.legend()
    plt.show()
    plt.close()


