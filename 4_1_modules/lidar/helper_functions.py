"""
Helper functions for LiDAR visualization and extent mapping.
Provides utilities for displaying LAS file spatial extents on interactive maps.
"""

import geopandas as gpd
import pandas as pd
import folium
import os
from .las import LAS

def display_las_extent(file_paths):
    """
    Display spatial extent of LAS files on an interactive Folium map.
    
    Args:
        file_paths: Single file path string or list of file paths
        
    Returns:
        Interactive Folium map showing LAS file extents
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    extents = []
    crs_list = []
    crs_set = set()
    file_names = []

    for file_path in file_paths:
        las = LAS(file_path)

        # Get extent using the LAS class
        extent_geom = las.geom
        extents.append(extent_geom)

        # Get CRS using the LAS class
        crs = las.crs
        crs_list.append(crs)
        crs_set.add(crs)

        # Get file basename
        file_names.append(os.path.basename(file_path))

    if len(crs_set) > 1:
        print("⚠️ Warning: Multiple CRSs detected. All extents will be transformed to WGS84.")

    # Transform each extent to WGS84
    gdf_list = []
    for extent, crs in zip(extents, crs_list):
        gdf = gpd.GeoDataFrame({'geometry': [extent]}, crs=f"EPSG:{crs}")
        gdf = gdf.to_crs(epsg=4326)
        gdf_list.append(gdf)

    combined_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))

    # Get original bounds
    min_x, min_y, max_x, max_y = combined_gdf.total_bounds

    # Define zoom-out factor (e.g., 0.5 = 50% more space)
    zoom_out_factor = 0.5

    # Calculate center of the bounds
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Calculate half-width and half-height with padding
    half_width = (max_x - min_x) * (1 + zoom_out_factor) / 2
    half_height = (max_y - min_y) * (1 + zoom_out_factor) / 2
    # Create folium map
    m = folium.Map(location=[center_y, center_x], zoom_start=13, max_zoom=30)

    for geom, file_name in zip(combined_gdf.geometry, file_names):
        folium.Rectangle(
            bounds=[(geom.bounds[1], geom.bounds[0]), (geom.bounds[3], geom.bounds[2])],
            color='blue',
            fill=True,
            fill_opacity=0.2,
            popup=file_name
        ).add_to(m)

    # Adjust zoom to fit all extents
    m.fit_bounds([
        [center_y - half_height, center_x - half_width],
        [center_y + half_height, center_x + half_width]
    ])

    return m