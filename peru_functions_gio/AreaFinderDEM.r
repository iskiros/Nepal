########################################################################################
# # Install required packages
# install.packages(c("tidyverse", "raster", "sf", "whitebox", "tmap", "stars", "rayshader", "rgl"))
# install.packages("hook_webgl")
# install.packages("devtools")
# install.packages("readxl")
# devtools::install_github("username/hook_webgl")
# install.packages("sf")

## Print R version
#print(R.version.string)
########################################################################################


########################################################################################

# Load required libraries
library(tidyverse)
library(raster)
library(sf)
library(whitebox)
library(tmap)
library(stars)
library(rayshader)
library(rgl)
library(readxl)
library(sf)
library(dplyr)

########################################################################################

##### Following instructions from: #####
### https://vt-hydroinformatics.github.io/rgeowatersheds.html ###
#### This is for 16, but part 15 explains why you need to change your DEM like so ####

########################################################################################



####### DEM PRE-PROCESSING #######

# Initialize whitebox toolbox
whitebox::wbt_init()

# Set theme
theme_set(theme_classic())

# Set tmap mode
tmap_mode("view")

# Read DEM raster
dem <- raster("/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/Raster.tif", crs = '+init=EPSG:4326')

# Write DEM raster with correct CRS
writeRaster(dem, "/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/Raster_crs.tif", overwrite = TRUE)

# Set negative values in DEM to NA
dem[dem < 0] <- NA

########################################################################################





##### DEM PLOTTING #####

# Plot DEM
tm_shape(dem) +
    tm_raster(style = "cont", palette = "PuOr", legend.show = TRUE) +
    tm_scale_bar()


########################################################################################





##### HILLSHADE AND AZIMUTH #####

# Generate hillshade
wbt_hillshade(dem = "/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/Raster_crs.tif",
                            output = "/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/Hillshade.tif",
                            azimuth = 115)


# Read hillshade raster
hillshade <- raster("/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/Hillshade.tif")

# Plot hillshade
tm_shape(hillshade) +
    tm_raster(style = "cont", palette = "-Greys", legend.show = FALSE) +
    tm_scale_bar()

########################################################################################    






##### BREACH DEPRESSIONS AND FILL #####

# Breach depressions in DEM
wbt_breach_depressions_least_cost(
    dem = "/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/Raster_crs.tif",
    output = "/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/Raster_breached.tif",
    dist = 5,
    fill = TRUE)

# Fill depressions in breached DEM
wbt_fill_depressions_wang_and_liu(
    dem = "/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/Raster_breached.tif",
    output = "/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/Raster_filled_breached.tif"
)

########################################################################################





##### PERFORM SIMULATION OF FLOW #####

# Perform D8 flow accumulation
wbt_d8_flow_accumulation(
    input = "/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/Raster_filled_breached.tif",
    output = "/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/FlowAcc.tif"
)

# Perform D8 pointer
wbt_d8_pointer(
    dem = "/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/Raster_filled_breached.tif",
    output = "/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/FlowDir.tif"
)

########################################################################################







##### IMPORT POUR POINTS FROM EXCEL OF OUR PROCESED POINTS #####

# Import pour points from Excel
pour_points <- read_excel("/Users/enrico/PeruRiversProject/chemweathering/data/fraction_data.xlsx", sheet = 1)

# Convert pour points to spatial points
ppointsSP <- sf::st_as_sf(pour_points, coords = c("longitude_converted", "latitude_converted"), crs = 4326)

# Abbreviate column names because if not it is too long. Keep in mind this is not ideal as it can lead to confusion.
# Make sure to compare the original column names with the abbreviated ones.

names(ppointsSP) <- abbreviate(names(ppointsSP), minlength = 10)

# Round numeric columns
ppointsSP <- ppointsSP %>%
    mutate(across(where(is.numeric), ~ round(., digits = 6)))

# Write pour points as shapefile
st_write(ppointsSP, "/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/pourpoints.shp", delete_layer = TRUE)

########################################################################################




##### EXTRACT STREAMS AND SNAP POUR POINTS TO SAID STREAMS WITH JENSON #####

# Extract streams from flow accumulation raster
wbt_extract_streams(flow_accum = "/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/FlowAcc.tif",
                                        output = "/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/Raster_Streams.tif",
                                        threshold = 6000)

# Snap pour points to stream network
wbt_jenson_snap_pour_points(pour_pts = "/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/pourpoints.shp",
                                                        streams = "/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/Raster_Streams.tif",
                                                        output = "/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/Snapped_PourPoints.shp",
                                                        snap_dist = 0.0005)


wbt_raster_streams_to_vector(streams = "/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/Raster_Streams.tif",
                             d8_pntr = "/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/FlowDir.tif",
                             output = "/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/Streams.shp")


streams <- st_read("/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/Streams.shp")

########################################################################################





##### SHOW STREAMS #####

tm_shape(hillshade)+
  tm_raster(style = "cont",palette = "-Greys", legend.show = FALSE)+
tm_shape(streams)+
  tm_lines(col = "blue")+
  tm_scale_bar()



######################################################################################## 




##### READ POINTS AND PLOT THEM WITH STREAMS #####

# Read snapped pour points shapefile
pp <- shapefile("/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/Snapped_PourPoints.shp")

# Read streams raster
streams <- raster("/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/Raster_Streams.tif")

# Plot streams and pour points
tm_shape(streams) +
    tm_lines(col = "blue")+
    tm_scale_bar() +
tm_shape(pp) +
    tm_dots(col = "red")

########################################################################################





##### WATERSHED DELINEATION #####

# Perform watershed delineation
wbt_watershed(d8_pntr = "/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/FlowDir.tif",
                            pour_pts = "/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/pourpoints.shp",
                            output = "/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/Brush_Watersheds.tif")

# Read watershed raster
ws <- raster("/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/Brush_Watersheds.tif")

# Add watershed number to pour points
pp$brush_watersheds <- raster::extract(ws, pp)

########################################################################################




##### PLOT WATERSHEDS #####

# Plot hillshade, watersheds, and pour points
tm_shape(hillshade) +
    tm_raster(style = "cont", palette = "-Greys", legend.show = FALSE) +
tm_shape(ws) +
    tm_raster(legend.show = TRUE, alpha = 0.5, style = "pretty") +
tm_shape(pp) +
    tm_dots(col = "red") +
    tm_text(text = "brush_watersheds", col = "black", size = 0.8)


########################################################################################




##### EXPORT POUR POINTS WITH WATERSHEDS TO CSV #####

# Convert pour points to dataframe
pp_df <- st_as_sf(pp) %>%
    mutate(
        latitude = st_coordinates(.)[,2],
        longitude = st_coordinates(.)[,1]
    ) %>%
    st_drop_geometry()

# Write pour points dataframe to CSV
write.csv(pp_df, "/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/pourpoints_with_watersheds.csv", row.names = FALSE)

########################################################################################



##### PLOT HILLSHADE, WATERSHED AND POINTS AGAIN #####

# Plot hillshade, watersheds, and pour points
tm_shape(hillshade) +
    tm_raster(style = "cont", palette = "-Greys", legend.show = FALSE) +
tm_shape(ws) +
    tm_raster(legend.show = TRUE, alpha = 0.5, style = "pretty") +
tm_shape(pp) +
    tm_dots(col = "red")

########################################################################################






##### CALCULATE AREA FOR EACH WATERSHED, IN CORRECT COORDINATES #####

# Load the watershed shapefile
watershed <- st_read("/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/ws1shp.shp")

# Ensure the CRS is projected (e.g., UTM)
watershed_proj <- st_transform(watershed, crs = 32718) # 32718 for southern hem

# Calculate the area of each watershed in square meters
watershed_proj <- watershed_proj %>%
  mutate(area_m2 = st_area(watershed_proj))

# Convert area to numeric for further calculations
watershed_proj <- watershed_proj %>%
  mutate(area_m2 = as.numeric(area_m2))

# Add elevation data for each watershed polygon
watershed_proj <- watershed_proj %>%
  mutate(elevation = raster::extract(dem, watershed_proj))

# Load the pour points shapefile
pp <- st_read("/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/pourpoints.shp")

# Check the CRS of both objects
print(st_crs(pp))
print(st_crs(watershed_proj))

# If they do not match, transform the CRS of `pp` to match `watershed_proj`
if (st_crs(pp) != st_crs(watershed_proj)) {
  pp <- st_transform(pp, crs = st_crs(watershed_proj))
}

# Now perform the spatial join
pp_watershed <- st_join(pp, watershed_proj, join = st_intersects)





##### CONVERT TO DATAFRAME AND EXPORT TO CSV #####

# Make a dataframe with the area of each watershed in square meters
area_df <- pp_watershed %>%
  group_by(unique_cod) %>%
  summarize(total_area_m2 = sum(area_m2, na.rm = TRUE))

print(area_df)

# rename column unique_cod to unique code
area_df <- area_df %>%
  rename(unique_code = unique_cod)

print(area_df)


library(dplyr)

# Try selecting columns again
#area_df <- area_df %>%
#  dplyr::select(unique_code, total_area_m2)

print(area_df)


# Export area_df to a CSV file
write.csv(area_df, "/Users/enrico/Desktop/ROKOS INTERNSHIP/QGIS/area_df.csv", row.names = FALSE)

########################################################################################




# RULES FOR LATER #

###### Now cumulatively add the area of each watershed to the next one in the sequence,
#if the name begins with R, starting with the highest elevation watershed.
#If the name begins with T, keep the original area.
#The final area for each watershed should be stored in a new column called final_area_m2.
#The total area for each unique code should be summarized in a new dataframe called area_df.
#Print the area_df dataframe to the console.


