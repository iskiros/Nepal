import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

import os


#- Unfinished code regarding a DEM model import. Unused in the final version of the project.


dem_path = '/Users/enrico/PeruRiversProject/chemweathering/data/Raster.tif'
output = os.getcwd() + dem_path

#elevation.clip(bounds=bounds, output=output, product='SRTM3')
