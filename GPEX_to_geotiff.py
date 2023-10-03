#!/usr/bin/env python

# This file converts GPEX data to geotiff.

"""
GPEX extreme precipitation data in geotiff format for Haliacmon river drainage basin.

GPEX_RT_10.tiff    GPEX_RT_2.tiff     GPEX_RT_39.tiff    GPEX_RT_500.tiff
GPEX_RT_100.tiff   GPEX_RT_20.tiff    GPEX_RT_5.tiff
GPEX_RT_1000.tiff  GPEX_RT_200.tiff   GPEX_RT_50.tiff

Each file contains data for one return period. Together there are ten files
for return periods for years
[2, 5, 10, 20, 39, 50, 100, 200, 500, 1000]
Inside each file there are eight bands that corresponds to event durations
of the following hours:
[3, 6, 12, 24, 48, 72, 120, 240]

The value is extreme precipitation return levels in mm estimated using the
Generalized Extreme Value (GEV) distribution.

The data is restricted to a 14 x 12 grid that contains Greek Haliacmon river drainage basin.
xmin, xmax, ymin, ymax = 20.75, 22.05, 39.75, 40.85

The GPEX data set is described in article:
Extreme Precipitation Return Levels for Multiple Durations on a Global Scale,
Gründemann et al., 2021.
https://doi.org/10.1016/j.jhydrol.2023.129558
"""

import xarray as xr
import geopandas as gpd  # for reading the bounding box from shape file.
# import rioxarray  # need to have this installed for saving to geotiff

ds = xr.open_dataset('~/DATA/GPEX/GPEX.nc')

shape = gpd.read_file('Drainage_basin/basin.shp')
bb = shape.to_crs('EPSG:4326').bounds.values[0]
bb = bb + [-0.1, -0.1, 0.1, 0.1]

ds2 = ds.sel(lon=slice(bb[0], bb[2]), lat=slice(bb[3], bb[1]))

# return times
for tr in ds['tr'].values:
    file = f'/tmp/GPEX_RT_{tr}.tiff'
    rl = ds2['gev_estimate'].sel(tr=tr)    
    rl.transpose('dur', 'lat', 'lon').rio.to_raster(file, compute=True)
    print(f'wrote {file}')

