import sys
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, TwoSlopeNorm, Normalize
from matplotlib.lines import Line2D
import geodatasets
import geopandas as gpd
import re
import folium
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
import earthpy.spatial as es
import matplotlib.colors as mcolors
import matplotlib
import elevation
from shapely.geometry import Point, box, Polygon, MultiPolygon
from scipy.interpolate import RegularGridInterpolator
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import from_bounds
from scipy.interpolate import griddata

######### CODE USED FOR MAKING MAPS IN 2D OR 3D #########

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

def Bathymetry_Dual(df):
    
    # Copy the DataFrame to avoid modifying the original one
    df_copy = df.copy()
    
    ######### DIGITAL ELEVATION MODEL MANIPULATION #########
    
    dem_path = 'chemweathering/data/output_SRTMGL1.tif'
    output_path = os.path.join(os.getcwd(), dem_path)

    # Define bounds
    west, south, east, north = -75.32426908093115, -13.526283371232736, -76.57091918634246, -11.749558766395864
    bounds = (west - 0.05, south - 0.05, east + 0.05, north + 0.05)

    # Calculate the center of the bounds
    center_lon = (west + east) / 2
    center_lat = (south + north) / 2

    # Open the DEM raster file
    with rasterio.open(output_path) as dem_raster:
        src_crs = dem_raster.crs
        out_image, out_transform = mask(dem_raster, [box(*bounds)], crop=True)
        clipped_dem_array = out_image[0]

        dst_transform, width, height = calculate_default_transform(
            src_crs, 'EPSG:32718', clipped_dem_array.shape[1], clipped_dem_array.shape[0], *bounds
        )
        dst_array = np.zeros((height, width), dtype=np.float32)

        reproject(
            source=clipped_dem_array,
            destination=dst_array,
            src_transform=out_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs='EPSG:32718',
            resampling=Resampling.bilinear
        )

    dst_array[dst_array < 0] = np.nan
    print(f"Reprojected DEM shape: {dst_array.shape}")

    # Interpolating the DEM to plot
    x = np.linspace(bounds[0], bounds[2], dst_array.shape[1])
    y = np.linspace(bounds[1], bounds[3], dst_array.shape[0])
    x, y = np.meshgrid(x, y)
    z = dst_array
    z_flipped_rotated = np.flipud(np.fliplr(z))

    ######### IMPORTING DATA FOR GEOLOGY OVERLAY #########

    shapefile_path = "/Users/enrico/Downloads/Clipped Map Shapefile/Clipped_Map_Shapefile_New.shp"
    gdf = gpd.read_file(shapefile_path)

    shapefile_path2 = "/Users/enrico/Downloads/Canete shapefiles/Canete.shp"
    gdf2 = gpd.read_file(shapefile_path2)
    
    shapefile_path3 = "/Users/enrico/Desktop/ROKOS Internship/QGIS/clipped_hydrography.shp"
    gdf3 = gpd.read_file(shapefile_path3)
    
    shapefile_path4 = "/Users/enrico/Desktop/ROKOS Internship/QGIS/HydroRivers/HydroRIVERS_v10_sa_shp/HydroRIVERS_v10_sa_shp/Filtered_HydroRIVERS_v10_sa.shp"
    filtered_gdf4 = gpd.read_file(shapefile_path4)  

    ######### DEFINING COLORS AND LABELS FOR GEOLOGY OVERLAY #########     

    id_to_color = {
        1: 'cyan', 4: 'cyan', 5: 'cyan', 7: 'cyan', 8: 'cyan',
        2: 'red', 6: 'red',
        3: 'fuchsia', 9: 'fuchsia'
    }
    
    id_to_label = {
        1: 'sed', 4: 'sed', 5: 'sed', 7: 'sed', 8: 'sed',
        2: 'volcanic', 6: 'volcanic',
        3: 'plutonic', 9: 'plutonic'
    }
    
    priority_order = {6: 3, 7: 2, 1: 1, 4: 1, 5: 1, 8: 1, 2: 1, 3: 1, 9: 1}

    gdf['priority'] = gdf['id'].map(priority_order)
    gdf = gdf.sort_values(by='priority', ascending=True)

    def points_to_geodataframe(df, lon_col='longitude_converted', lat_col='latitude_converted'):
        geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
        gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        return gdf_points

    gdf_points = points_to_geodataframe(df_copy)

    def spatial_join_points_polygons(gdf_points, gdf_polygons, id_col='id'):
        gdf_points = gdf_points.to_crs(gdf_polygons.crs)
        joined_gdf = gpd.sjoin(gdf_points, gdf_polygons[[id_col, 'geometry']], how='left', predicate='within')
        joined_gdf = joined_gdf.drop(columns=['index_right'])
        return joined_gdf

    gdf_points_with_lithology = spatial_join_points_polygons(gdf_points, gdf)

    def assign_lithology_labels(gdf, id_to_label):
        gdf['lithology_label'] = gdf['id'].map(id_to_label)
        return gdf

    gdf_points_with_lithology = assign_lithology_labels(gdf_points_with_lithology, id_to_label)

    ######### PLOTTING THE MAPS SIDE BY SIDE #########

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ###### PLOT ON FIRST AXIS (ax1) ######

    c1 = ax1.contourf(x, y, z_flipped_rotated, cmap='Greys', alpha=0.7)

    plotted_labels = set()
    for geom, id_value in zip(gdf.geometry, gdf['id']):
        color = id_to_color.get(id_value, 'grey')
        label = id_to_label.get(id_value, '')
        if isinstance(geom, Polygon):
            x_poly, y_poly = geom.exterior.xy
            if label not in plotted_labels:
                ax1.fill(x_poly, y_poly, color=color, alpha=0.5, edgecolor='black', linewidth=0.5, label=label)
                plotted_labels.add(label)
            else:
                ax1.fill(x_poly, y_poly, color=color, alpha=0.5, edgecolor='black', linewidth=0.5)
        elif isinstance(geom, MultiPolygon):
            for part in geom.geoms:
                x_poly, y_poly = part.exterior.xy
                if label not in plotted_labels:
                    ax1.fill(x_poly, y_poly, color=color, alpha=0.5, edgecolor='black', linewidth=0.5, label=label)
                    plotted_labels.add(label)
                else:
                    ax1.fill(x_poly, y_poly, color=color, alpha=0.5, edgecolor='black', linewidth=0.5)

    gdf2.plot(ax=ax1, facecolor='none', edgecolor='yellow', alpha=1, linewidth=1, label='Canete Watershed')

    filtered_gdf4.plot(ax=ax1, facecolor='none', edgecolor='blue', alpha=0.7, linewidth=np.log10(filtered_gdf4['DIS_AV_CMS']), legend=True)

    markers = {'mainstream': 'o', 'tributary': '^', 'spring': '*'}

    vmin = df_copy['Net_Flux_Budget'].min()
    vmax = df_copy['Net_Flux_Budget'].max()

    cmap = 'seismic'
    norm = TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)

    for water_body, marker in markers.items():
        subset = df_copy[df_copy['water_body'] == water_body]
        sc1 = ax1.scatter(
            subset['longitude_converted'].values,
            subset['latitude_converted'].values,
            c=subset['Net_Flux_Budget'],
            cmap=cmap,
            s=100,
            alpha=0.9,
            edgecolor='black',
            marker=marker,
            label=water_body,
            norm=norm,
            linewidth=1
        )

    cbar_sc1 = fig.colorbar(sc1, ax=ax1, shrink=0.5, aspect=5, label='Net Flux Budget')
    ticks = np.linspace(vmin, vmax, num=9)
    cbar_sc1.set_ticks(ticks)
    cbar_sc1.set_ticklabels([f'{tick:.0f}' if tick != 0 else '0' for tick in ticks])

    lowermost_point = df_copy[df_copy['unique_code'] == 'RC02AWF-1121']
    net_flux_budget = lowermost_point['Net_Flux_Budget'].values[0]

    longitude = float(lowermost_point['longitude_converted'].values[0])
    latitude = float(lowermost_point['latitude_converted'].values[0])

    ax1.text(longitude + 1.1, latitude - 0.25, 
            f'RCO2 Specific Flux: \n{net_flux_budget}', 
            fontsize=12, color='red', ha='center')

    ax1.set_xlim(np.min(x), np.max(x))
    ax1.set_ylim(np.min(y), np.max(y))

    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Net_Flux_Budget in kg CO2/km^2/yr')





    ###### PLOT ON SECOND AXIS (ax2) ######

    c2 = ax2.contourf(x, y, z_flipped_rotated, cmap='Greys', alpha=0.7)

    for geom, id_value in zip(gdf.geometry, gdf['id']):
        color = id_to_color.get(id_value, 'grey')
        label = id_to_label.get(id_value, '')
        if isinstance(geom, Polygon):
            x_poly, y_poly = geom.exterior.xy
            ax2.fill(x_poly, y_poly, color=color, alpha=0.5, edgecolor='black', linewidth=0.5)
        elif isinstance(geom, MultiPolygon):
            for part in geom.geoms:
                x_poly, y_poly = part.exterior.xy
                ax2.fill(x_poly, y_poly, color=color, alpha=0.5, edgecolor='black', linewidth=0.5)

    gdf2.plot(ax=ax2, facecolor='none', edgecolor='yellow', alpha=1, linewidth=1, label='Canete Watershed')

    filtered_gdf4.plot(ax=ax2, facecolor='none', edgecolor='blue', alpha=0.7, linewidth=np.log10(filtered_gdf4['DIS_AV_CMS']), legend=True)

    vmin2 = df_copy['Net_Long_Term_Budget'].min()
    vmax2 = df_copy['Net_Long_Term_Budget'].max()

    cmap2 = 'Reds'

    for water_body, marker in markers.items():
        subset = df_copy[df_copy['water_body'] == water_body]
        sc2 = ax2.scatter(
            subset['longitude_converted'].values,
            subset['latitude_converted'].values,
            c=subset['Net_Long_Term_Budget'],
            cmap=cmap2,
            s=100,
            alpha=0.9,
            edgecolor='black',
            marker=marker,
            label=water_body,
            norm=Normalize(vmin=vmin2, vmax=vmax2),
            linewidth=1
        )

    cbar_sc2 = fig.colorbar(sc2, ax=ax2, shrink=0.5, aspect=5, label='Net Long Term kg-CO2/yr')
    
    ax2.set_xlim(np.min(x), np.max(x))
    ax2.set_ylim(np.min(y), np.max(y))

    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title('Net Long Term kg-CO2/yr')

    #plt.show()

    plt.close(fig)
    
    # ### HEATMAP ###
        
    # # Plot the watershed boundaries on ax2
    # gdf2.plot(ax=ax2, facecolor='none', edgecolor='black', alpha=1, linewidth=1, label='Canete Watershed', zorder=2)

    # cmap2 = 'seismic'

    # # Define the grid size (you can adjust this for more or less detail)
    # grid_size = 100

    # # Create grid points where you want to interpolate data
    # grid_lon = np.linspace(np.min(df_copy['longitude_converted']), np.max(df_copy['longitude_converted']), grid_size)
    # grid_lat = np.linspace(np.min(df_copy['latitude_converted']), np.max(df_copy['latitude_converted']), grid_size)
    # grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

    # # Interpolate the data
    # grid_z = griddata(
    #     (df_copy['longitude_converted'], df_copy['latitude_converted']),
    #     df_copy['Net_Long_Term_Budget'],
    #     (grid_lon, grid_lat),
    #     method='cubic'
    # )

    # vmin2 = df_copy['Net_Long_Term_Budget'].min()
    # vmax2 = df_copy['Net_Long_Term_Budget'].max()

    # #cmap2 = 'coolwarm'
    # #norm = TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)

    # # Use TwoSlopeNorm to center the colormap at the adjusted vcenter
    # norm = TwoSlopeNorm(vmin=vmin2, vmax=vmax2, vcenter=0)

    # # Plot the interpolated heatmap on ax2 using the centered normalization
    # c2 = ax2.contourf(grid_lon, grid_lat, grid_z, levels=100, cmap=cmap2, norm=norm)


    # # Overlay the original scatter points on top of the heatmap
    # markers = {'mainstream': 'o', 'tributary': '^', 'spring': '*'}

    # for water_body, marker in markers.items():
    #     subset = df_copy[df_copy['water_body'] == water_body]
    #     sc2 = ax2.scatter(
    #         subset['longitude_converted'].values,
    #         subset['latitude_converted'].values,
    #         c=subset['Net_Long_Term_Budget'],
    #         cmap=cmap2,
    #         s=100,
    #         alpha=0.9,
    #         edgecolor='black',
    #         marker=marker,
    #         norm=norm,
    #         linewidth=1,
    #         zorder=3  # Ensure points are drawn on top
    #     )



    # # Add a colorbar for the heatmap, centered on the appropriate value
    # cbar_sc2 = fig.colorbar(c2, ax=ax2, shrink=0.5, aspect=5, label='Net Long Term kg-CO2/yr')

    # ax2.set_xlim(np.min(x), np.max(x))
    # ax2.set_ylim(np.min(y), np.max(y))

    # ax2.set_xlabel('Longitude')
    # ax2.set_ylabel('Latitude')
    # ax2.set_title('Net Long Term kg-CO2/yr Heatmap')

    # plt.show()
    # plt.close(fig)
    
    
    #return gdf_points_with_lithology
