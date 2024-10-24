import sys
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import geodatasets
import geopandas as gpd
import re
import folium
from folium.plugins import HeatMap
import earthpy as et
import webbrowser
import statsmodels.api as sm
import elevation
import shapely.geometry
import seaborn as sns
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
import rasterio
import webbrowser
import earthpy.spatial as es
import matplotlib.colors as mcolors
import matplotlib
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
from matplotlib.colors import ListedColormap, Normalize
from scipy.interpolate import griddata


######### CODE USED FOR MAKING MAPS IN 2D OR 3D #########


######### CODE USED FOR ROTATING THE POINTS FOR THE MAP BECAUSE THE RASTER AND THE POINTS ARE NOT FITTED #########

def rotate_points_180(df_lons, df_lats, center_lon, center_lat):
    # Rotate points by 180 degrees around the center
    rotated_lons = 2 * center_lon - df_lons
    rotated_lats = 2 * center_lat - df_lats
    return rotated_lons, rotated_lats


def get_gradient_color(Fcarb, Fsulf, gradient, gradient_size):
    Fcarb = np.clip(Fcarb, 0, 1)
    Fsulf = np.clip(Fsulf, 0, 1)
    
    i = int((Fcarb + Fsulf) / 2 * (gradient_size - 1))  # Average to find position in gradient
    color = gradient[i]
    hex_color = '#{:02x}{:02x}{:02x}'.format(int(color[0]*255), int(color[1]*255), int(color[2]*255))
    return hex_color


def get_gradient_index(Fcarb, Fsulf, gradient_size):
    Fcarb = np.clip(Fcarb, 0, 1)
    Fsulf = np.clip(Fsulf, 0, 1)
    index = (Fcarb + Fsulf) / 2  # This gives a normalized value between 0 and 1
    return index

######### CODE USED FOR PLOTTING THE MAP #########


def Bathymetry_Heatmap(df):
    
    ######### IMPORT DATAFRAME #########
    
    # Copy the DataFrame to avoid modifying the original one
    df_copy = df.copy()
    
    ####################################
    
    df_copy.to_excel('chemweathering/data/testingspecificflux.xlsx')
    
    
    ######### DIGITAL ELEVATION MODEL MANIPULATION #########
    
    # Define the path to the DEM (Digital Elevation Model) raster file
    dem_path = 'chemweathering/data/output_SRTMGL1.tif'
    
    output_path = os.path.join(os.getcwd(), dem_path)  # Adjusted path output

    # Define bounds
    west, south, east, north = -75.32426908093115, -13.526283371232736, -76.57091918634246, -11.749558766395864
    bounds = (west - 0.05, south - 0.05, east + 0.05, north + 0.05)

    # Calculate the center of the bounds
    center_lon = (west + east) / 2
    center_lat = (south + north) / 2

    # Open the DEM raster file
    with rasterio.open(output_path) as dem_raster:
        src_crs = dem_raster.crs
        
        # Clip the DEM to the specified bounds
        out_image, out_transform = mask(dem_raster, [shapely.geometry.box(*bounds)], crop=True)
        clipped_dem_array = out_image[0]

        # Calculate the destination transform based on the bounds and the shape of the clipped array
        dst_transform = from_bounds(*bounds, clipped_dem_array.shape[1], clipped_dem_array.shape[0])
        
        # Define the destination CRS
        dst_crs = 'EPSG:32718'

        # Get the dimensions of the clipped DEM
        height, width = clipped_dem_array.shape

        # Create the destination transform and array
        dst_transform, width, height = calculate_default_transform(
            src_crs, dst_crs, width, height, *bounds
        )
        dst_array = np.zeros((height, width), dtype=np.float32) #define x and y and z

        # Reproject the source data to the destination CRS
        reproject(
            source=clipped_dem_array,
            destination=dst_array,
            src_transform=out_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )

    # Filter out elevation values below 0
    dst_array[dst_array < 0] = np.nan # nice shorthand
    print(f"Reprojected DEM shape: {dst_array.shape}")
    
    ######################################################
    
    
    
    
    ######### INTERPOLATING THE DEM TO PLOT #########

    # Generate the x and y coordinates in the projected CRS
    x = np.linspace(bounds[0], bounds[2], dst_array.shape[1])
    y = np.linspace(bounds[1], bounds[3], dst_array.shape[0])
    x, y = np.meshgrid(x, y)
    z = dst_array

    # Interpolator for DEM elevations
    interpolator = RegularGridInterpolator((y[:, 0], x[0, :]), z, bounds_error=False, fill_value=np.nan)

    # Ensure the DataFrame coordinates are in the same CRS as the DEM
    df_lons = df_copy['longitude_converted'].values
    df_lats = df_copy['latitude_converted'].values
    
    
    
    
    ######### ROTATE THE POINTS BECAUSE THE RASTER AND THE POINTS ARE OFF BY 180 DEGREES #########

    # Rotate the points by 180 degrees around the center
    rotated_lons, rotated_lats = rotate_points_180(df_lons, df_lats, center_lon, center_lat)

    # Adjust altitude values to be on or slightly above the DEM surface
    #points = np.array([rotated_lats, rotated_lons]).T #for when plotting in 3D
    
    points = np.array([df_lats, df_lons]).T
    
    
    
    
    
    ######### ALTITUDES FOR 3D PLOT. SINCE DISCONTINUED #########
    
    dem_alts = interpolator(points)  # Get DEM altitudes at the points
    df_alts = dem_alts + 2000  # Adding 500 meters to ensure points are above the surface
    
    ###############################################################
    
    z_flipped_rotated = np.flipud(np.fliplr(z))
    
    #############################################



    ######### IMPORTING DATA FOR GEOLOGY OVERLAY #########

    # Load the shapefile
    shapefile_path = "/Users/enrico/Downloads/Clipped Map Shapefile/Clipped_Map_Shapefile_New.shp"
    gdf = gpd.read_file(shapefile_path)
    
    
    
    
    
    ######### OVERLAYING CATCHMENT AND HYDROLOGY DATA, EDITED ON QGIS #########
    
    # Load second shapefile from Canete
    shapefile_path2 = "/Users/enrico/Downloads/Canete shapefiles/Canete.shp"
    gdf2 = gpd.read_file(shapefile_path2)
    
    
    # Load third shapefile for river hydrology
    shapefile_path3 = "/Users/enrico/Desktop/ROKOS Internship/QGIS/clipped_hydrography.shp"
    gdf3 = gpd.read_file(shapefile_path3)
    
    ######### I DO NOT THINK WE USE SHAPEFILE 3 ANYMORE, rather the hydrology shapefile from HYDRORivers #########
            
    shapefile_path4 = "/Users/enrico/Desktop/ROKOS Internship/QGIS/HydroRivers/HydroRIVERS_v10_sa_shp/HydroRIVERS_v10_sa_shp/Filtered_HydroRIVERS_v10_sa.shp"
    filtered_gdf4 = gpd.read_file(shapefile_path4)  

    ########################################################################

    
         
         
    ######### DEFINING COLORS AND LABELS FOR GEOLOGY OVERLAY #########     
            
    
    #For Shapefile
    # Define color mapping based on ID
    id_to_color = {
        1: 'cyan',
        4: 'cyan',
        5: 'cyan',
        7: 'cyan',
        8: 'cyan',
        2: 'red',
        6: 'red',
        3: 'fuchsia',
        9: 'fuchsia'
    }
    
    # #### JUST FOR HEATMAP ####
    
    # id_to_color = {
    #     1: 'white',
    #     4: 'white',
    #     5: 'white',
    #     7: 'white',
    #     8: 'white',
    #     2: 'white',
    #     6: 'white',
    #     3: 'white',
    #     9: 'white'
    # }
    
    # ############################

    # Define label mapping based on ID
    id_to_label = {
        1: 'sed',
        4: 'sed',
        5: 'sed',
        7: 'sed',
        8: 'sed',
        2: 'volcanic',
        6: 'volcanic',
        3: 'plutonic',
        9: 'plutonic'
    }
    
    # Define priority order (higher values will be plotted last and thus on top)
    priority_order = {6: 3, 7: 2, 1: 1, 4: 1, 5: 1, 8: 1, 2: 1, 3: 1, 9: 1} #just because some of them do not show

    # Sort the GeoDataFrame based on priority
    gdf['priority'] = gdf['id'].map(priority_order)
    gdf = gdf.sort_values(by='priority', ascending=True)

    ################################################################

        ######### Convert Points to a GeoDataFrame #########

    def points_to_geodataframe(df, lon_col='longitude_converted', lat_col='latitude_converted'):
        """
        Converts a DataFrame of points to a GeoDataFrame.

        Parameters:
        - df: The original DataFrame containing longitude and latitude.
        - lon_col: The name of the column for longitude values.
        - lat_col: The name of the column for latitude values.

        Returns:
        - gdf_points: A GeoDataFrame of points.
        """
        geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
        gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        return gdf_points

    # Convert DataFrame to GeoDataFrame
    gdf_points = points_to_geodataframe(df_copy)

    ######### Perform Spatial Join #########

    def spatial_join_points_polygons(gdf_points, gdf_polygons, id_col='id'):
        """
        Perform a spatial join to find which points lie within which polygons.

        Parameters:
        - gdf_points: The GeoDataFrame of points.
        - gdf_polygons: The GeoDataFrame of polygons (e.g., lithology shapefile).
        - id_col: The column in the polygon GeoDataFrame to use for lithology identification.

        Returns:
        - joined_gdf: A GeoDataFrame with points and their corresponding lithology ID.
        """
        # Ensure both GeoDataFrames have the same CRS
        gdf_points = gdf_points.to_crs(gdf_polygons.crs)

        # Perform the spatial join
        joined_gdf = gpd.sjoin(gdf_points, gdf_polygons[[id_col, 'geometry']], how='left', predicate='within')

        # Remove duplicates or unnecessary columns if needed
        joined_gdf = joined_gdf.drop(columns=['index_right'])

        return joined_gdf

    # Perform spatial join
    gdf_points_with_lithology = spatial_join_points_polygons(gdf_points, gdf)

    ######### Assign Lithology Labels #########

    def assign_lithology_labels(gdf, id_to_label):
        """
        Assign lithology labels to points based on their location within polygons.

        Parameters:
        - gdf: The GeoDataFrame with points and lithology IDs.
        - id_to_label: A dictionary mapping IDs to lithology labels.

        Returns:
        - gdf: The updated GeoDataFrame with lithology labels.
        """
        gdf['lithology_label'] = gdf['id'].map(id_to_label)
        return gdf

    # Assign lithology labels to points
    gdf_points_with_lithology = assign_lithology_labels(gdf_points_with_lithology, id_to_label)

    
    #print(gdf_points_with_lithology.head())


    ######### PLOTTING THE MAP #########

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))



    ###### UNCOMMENT IF YOU WANT THE DEM MODEL ######

    # Plot the DEM data in grayscale
    c = ax.contourf(x, y, z_flipped_rotated, cmap='Greys', alpha=0.7)
    
    ########################################################
    
    
    

    # Plot the overlay shapefile without colours - GENERAL CODE FOR PLOTTING A SHAPEFILE
    #gdf.plot(ax=ax, facecolor='none', edgecolor='blue', alpha=0.7, linewidth=1)
    
    


    
    
    ######### GEOLOGY OVERLAY #########
    
    
    # Plot the overlay shapefile with specified colors and labels
    plotted_labels = set()  # Keeps track of which labels have been plotted
    for geom, id_value in zip(gdf.geometry, gdf['id']):
        color = id_to_color.get(id_value, 'grey')  # Default to grey if ID is not in the mapping
        label = id_to_label.get(id_value, '')
        if isinstance(geom, Polygon): ## This change ensures that geom is explicitly checked to be a Polygon type using the isinstance function, which is more Pythonic and handles the type check more explicitly.
            x_poly, y_poly = geom.exterior.xy  # Extract the exterior coordinates (x and y) of the polygon
            if label not in plotted_labels:
                ax.fill(x_poly, y_poly, color=color, alpha=0.5, edgecolor='black', linewidth=0.5, label=label)  # Fill the polygon with the specified color and add a label
                plotted_labels.add(label)
            else:
                ax.fill(x_poly, y_poly, color=color, alpha=0.5, edgecolor='black', linewidth=0.5)  # Fill the polygon without adding a label if it has already been plotted
        elif isinstance(geom, MultiPolygon):
            for part in geom.geoms: #  In the original code, geom is treated as an iterable directly, which caused the TypeError. The revised code accesses the .geoms attribute of a MultiPolygon, which is an iterable of its constituent polygons, ensuring proper iteration.
                x_poly, y_poly = part.exterior.xy
                if label not in plotted_labels:
                    ax.fill(x_poly, y_poly, color=color, alpha=0.5, edgecolor='black', linewidth=0.5, label=label)
                    plotted_labels.add(label)
                else:
                    ax.fill(x_poly, y_poly, color=color, alpha=0.5, edgecolor='black', linewidth=0.5)

    #############################################################





    ######### PLOTTING THE CATCHMENT AND HYDROLOGY DATA #########

    # Plot second shapefile:
    gdf2.plot(ax=ax, facecolor='none', edgecolor='yellow', alpha=1, linewidth=1, label='Canete Watershed')
    
    # Convert to GeoJSON format
    gdf2_geojson = gdf2.to_json()
    
    #Plot third shapefile: - the Peruvian Hydrology
    #gdf3.plot(ax=ax, facecolor='none', edgecolor='grey', alpha=0.8, linewidth=0.3)
    
    ## Make the shapefile thicker where DIS_Av_CMS in filtered_gdf4 is greatest and smaller when it is smallest, and add a legend:
    ## Get vmin and vmax for DIS_AV_CMS
    vmin, vmax = filtered_gdf4['DIS_AV_CMS'].min(), filtered_gdf4['DIS_AV_CMS'].max()
    
    #print('Minimum Discharge:', vmin, '\n Maximum discharge:', vmax)
    
    #Minimum Discharge: 0.014  m^3/s
    #Maximum discharge: 275.559 m^3/s
    
    # Plot the fourth shapefile with varying linewidth based on DIS_AV_CMS with a log scale between vmin and vmax:
    filtered_gdf4.plot(ax=ax, facecolor='none', edgecolor='blue', alpha=0.7, linewidth=np.log10(filtered_gdf4['DIS_AV_CMS']), legend=True)
    
    ## Add a scale bar for the discharge values, remember it is log form. Scale bar should report widths, not colour:
    # What is the width of the largest line?
    #print('Width of largest line:', np.log10(vmax))
    
    
   
    ############### PLOTTING THE SCATTER POINTS FOR ANYTHING WITH WATER BODY MARKERS ###############


    
    # Define markers for different water body types
    markers = {'mainstream': 'o', 'tributary': '^', 'spring': '*'}

    # Remove any absurdly negative values before finding vmin and vmax
    vmin = df_copy['Net_Flux_Budget'].min()
    vmax = df_copy['Net_Flux_Budget'].max()

    # Use the 'seismic' colormap for a diverging color scheme
    cmap = 'seismic'

    # Create a custom normalization that centers at zero
    norm = TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)

    # Create a scatter plot using the custom normalization
    for water_body, marker in markers.items():
        subset = df_copy[df_copy['water_body'] == water_body]
        sc = ax.scatter(
            subset['longitude_converted'].values,
            subset['latitude_converted'].values,
            c=subset['Net_Flux_Budget'],
            cmap=cmap,
            s=100,
            alpha=0.9,
            edgecolor='black',
            marker=marker,
            label=water_body,
            norm=norm,  # Apply the custom normalization
            linewidth=1
        )

    # Add a colorbar with the custom normalization
    cbar_sc = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5, label='Net Flux Budget')

    # Adjust ticks on the colorbar to ensure they reflect the symmetric scaling
    ticks = np.linspace(vmin, vmax, num=9)
    cbar_sc.set_ticks(ticks)
    cbar_sc.set_ticklabels([f'{tick:.0f}' if tick != 0 else '0' for tick in ticks])
    
    # Display the Net Flux Budget at the lowermost point (that being RC02AWF-1121) on the graph:
    # Ensure the coordinates are scalars, not single-element Series
    lowermost_point = df_copy[df_copy['unique_code'] == 'RC02AWF-1121']
    net_flux_budget = lowermost_point['Net_Flux_Budget'].values[0]

    # Convert longitude and latitude to scalar floats
    longitude = float(lowermost_point['longitude_converted'].values[0])
    latitude = float(lowermost_point['latitude_converted'].values[0])

    # Display the Net Flux Budget at the lowermost point
    print('RC02AWF-1121, Specific Flux =', net_flux_budget)
    ax.text(longitude + 1.2, latitude - 0.25, 
            f'RCO2 Specific Flux: \n{net_flux_budget}', 
            fontsize=12, color='red', ha='center')
    
    
    # Set the title to reflect the centered colorbar
    #ax.set_title('Net_Flux_Budget with Zero-Centered Colorbar')

    

    # Set axis limits to match the DEM extent
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))

    # Labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Manually get unique handles and labels without using ax.get_legend_handles_labels()
    unique_handles_labels = {}

    # Add custom scatter markers to the legend for each water body type
    for water_body, marker in markers.items():
        custom_scatter = Line2D([0], [0], marker=marker, color='black', linestyle='None', label=water_body)
        unique_handles_labels[water_body] = custom_scatter

    # Add other custom handles if necessary, e.g., Canete Watershed
    custom_handle = Line2D([0], [0], color='yellow', lw=2, label='Canete Watershed')
    unique_handles_labels['Canete Watershed'] = custom_handle

    # Add the custom legend to the plot
    ax.legend(loc='upper left', handles=unique_handles_labels.values())
                                                            

    
    # [0] and [0] represent the x and y coordinates of a point on a line.

    # The plt.Line2D() function is used to create a Line2D object, which represents a line in a plot. 
    # The first argument [0] specifies the x-coordinates of the line, and the second argument [0] specifies the y-coordinates of the line.

    # In this specific case, the [0] values indicate that the line consists of a single point at the origin (0, 0) in the plot. 
    # This means that the line will not be visible, as it only represents a single point.

    # If you want to create a line with multiple points, you can provide a list of x-coordinates and y-coordinates as arguments to plt.Line2D(). 
    # For example, plt.Line2D([0, 1, 2], [0, 1, 2]) would create a line with three points: (0, 0), (1, 1), and (2, 2).
    
    # This line creates a custom legend entry for the "Canete Watershed" by creating a Line2D object with specific properties such as color, linewidth, and label. 
    # The custom_handle variable holds this custom legend entry. 
    # Then, it adds this custom handle to the unique_handles_labels dictionary with the key 'Canete Watershed'.


    #ax.legend(loc='upper left', handles=unique_handles_labels.values())
    
    ## add a title for this plot
    ax.set_title('Net_Flux_Budget in kg CO2/km^2/yr')


    # Save the figure
    plt.savefig('chemweathering/data/Net_Flux_Budget_Map.png', dpi=300)
    #plt.show()
    
    plt.close(fig)
    
    #################################################################################
    
    #### HEATMAP ####
    
        
    # 2. Create the Folium map
    map_center = [np.mean(df_copy['latitude_converted']), np.mean(df_copy['longitude_converted'])]
    hmap = folium.Map(location=map_center, zoom_start=10)

    folium.TileLayer('cartodbpositron').add_to(hmap)
    #folium.LayerControl().add_to(hmap)
    
    heat_data = list(zip(df_copy['latitude_converted'].values,
                         df_copy['longitude_converted'].values,
                         df_copy['Net_Long_Term_Budget'].values))

    HeatMap(heat_data, 
            min_opacity=0.4, 
            radius=17, 
            blur=15, 
            max_zoom=0.8,
            zorder = 2).add_to(hmap)
    
    # 4. Overlay the shapefile (GeoJSON)
    folium.GeoJson(
        gdf2_geojson,
        name="Canete Watershed",
        style_function=lambda feature: {
            'fillColor': 'gray',
            'color': 'gray',
            'weight': 2,
            'fillOpacity': 0.2,
        }
    ).add_to(hmap)

    # 4. Add a legend (as an HTML object) to the map
    legend_html = """
        <div style="position: fixed; 
        top: 50px; left: 50px; width: 200px; height: 150px; 
        background-color: white; z-index:9999; font-size:14px;
        border:2px solid grey; border-radius: 10px; padding: 10px;">
        <strong>Net Long Term Budget</strong> <br>
        <span style="background-color: blue; width: 20px; height: 10px; display: inline-block;"></span>&nbsp; Low <br>
        <span style="background-color: green; width: 20px; height: 10px; display: inline-block;"></span>&nbsp; Medium <br>
        <span style="background-color: red; width: 20px; height: 10px; display: inline-block;"></span>&nbsp; High - CO2 Release <br>
        </div>
        """
    
    hmap.get_root().html.add_child(folium.Element(legend_html))


    # 3. Save the Folium map as an HTML file
    folium_map_path = 'net_long_term_budget_map.html'
    hmap.save(folium_map_path)

    # 4. Create a combined HTML page
    combined_html_path = 'combined_map_and_plot.html'
    with open(combined_html_path, 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Combined Map and Plot</title>
            <style>
                .container {{
                    display: flex;
                    justify-content: space-between;
                }}
                .container > div {{
                    flex-basis: 48%;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div>
                    <h1>Net Flux Specific Budget </h1>
                    <img src="{'chemweathering/data/Net_Flux_Budget_Map.png'}" alt="Net Flux Budget Map" style="width:100%; max-width:800px;"/>
                </div>
                <div>
                    <h1>Net Long Term Budget kg CO2/yr</h1>
                    <iframe src="{folium_map_path}" width="100%" height="800"></iframe>
                </div>
            </div>
        </body>
        </html>
        """)

    # Optionally open the combined HTML in the default web browser
    webbrowser.open('file://' + os.path.realpath(combined_html_path))
    
    return gdf_points_with_lithology