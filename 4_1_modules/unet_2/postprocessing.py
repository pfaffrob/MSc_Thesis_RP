"""
U-Net postprocessing utilities for handling model predictions.
Includes functions for metadata copying and prediction merging.
"""

import os
import rasterio
import numpy as np
from rasterio.merge import merge

def copy_metadata(rgb_dir, gray_dir):
    """
    Copies metadata from RGB images to grayscale images and compresses them with LZW.

    Parameters:
    rgb_dir (str): Path to the directory containing the RGB images.
    gray_dir (str): Path to the directory containing the grayscale images.
    """

    # Get a list of all files in the directory
    rgb_files = os.listdir(rgb_dir)
    gray_files = os.listdir(gray_dir)

    # Iterate over the grayscale images
    for gray_file in gray_files:
        # Check if the corresponding RGB file exists
        if gray_file in rgb_files:
            # Open the RGB and grayscale images
            with rasterio.open(os.path.join(rgb_dir, gray_file)) as src:
                rgb_meta = src.meta

            with rasterio.open(os.path.join(gray_dir, gray_file)) as gray:
                gray_image = gray.read(1)

            # Update the metadata of the grayscale image
            rgb_meta.update({"driver": "GTiff",
                             "height": gray_image.shape[0],
                             "width": gray_image.shape[1],
                             "count": 1,
                             "compress": "lzw"})

            # Write the grayscale image with the new metadata
            with rasterio.open(os.path.join(gray_dir, gray_file), 'w', **rgb_meta) as dst:
                dst.write(gray_image, 1)
                


def merge_geotiffs(tiff_dir, output_path):
    """
    Merges GeoTIFF files, considering overlaps.

    Parameters:
    tiff_dir (str): Path to the directory containing the GeoTIFF files.
    output_path (str): Path to the output file.
    """

    # Get a list of all GeoTIFF files in the directory
    tiff_files = [os.path.join(tiff_dir, f) for f in os.listdir(tiff_dir) if f.endswith('.tif')]

    # Open all GeoTIFF files
    src_files_to_mosaic = [rasterio.open(f) for f in tiff_files]

    # Merge function returns a single mosaic array and the transformation info
    mosaic, out_trans = merge(src_files_to_mosaic)

    # Average the overlaps and assign 1 if average >= 0.5
    mosaic = np.where(mosaic >= 0.5, 1, 0)

    # Copy the metadata
    out_meta = src_files_to_mosaic[0].meta.copy()

    # Update the metadata
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "compress": "lzw"
    })

    # Write the mosaic raster to disk
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)