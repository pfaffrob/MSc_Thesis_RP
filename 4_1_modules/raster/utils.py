"""
Raster data processing utilities for GeoTIFF manipulation.
Includes functions for merging, clipping, reprojecting, and inspecting raster files.
"""

import os
import rasterio
from rasterio import warp
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from shapely.geometry import box
import pandas as pd

def inspect_geotiff_metadata(tif_path):
    """
    Print comprehensive metadata for a GeoTIFF file.
    
    Args:
        tif_path: Path to the GeoTIFF file
    """
    with rasterio.open(tif_path) as src:
        print("CRS:", src.crs)
        print("Transform:", src.transform)
        print("Width x Height:", src.width, "x", src.height)
        print("Band Count:", src.count)
        print("Data Type:", src.dtypes)
        print("NoData Value:", src.nodata)
        print("Compression:", src.profile.get("compress"))
        print("Driver:", src.driver)

        # Band descriptions and color interpretations
        for i in range(1, src.count + 1):
            print(f"Band {i} Description:", src.descriptions[i - 1])
            print(f"Band {i} ColorInterp:", src.colorinterp[i - 1])

        # Global tags
        print("Global Tags:", src.tags())

        # Per-band tags
        for i in range(1, src.count + 1):
            print(f"Band {i} Tags:", src.tags(i))

def get_num_bands(tif_path):
    """
    Returns the number of bands in a TIFF file.

    Parameters:
    tif_path (str): Path to the TIFF file.

    Returns:
    int: Number of bands.
    """
    with rasterio.open(tif_path) as src:
        return src.count


def merge_raster(input_paths, output_path, nodata=None):
    """
    Merge multiple raster files into one, reprojecting to EPSG:2193, removing alpha bands,
    preserving dtype and band descriptions, and compressing with LZW.

    Parameters:
        input_paths (list of str): Paths to input raster files.
        output_path (str): Path to save the merged output raster or directory to save 'merged.tif'.
        nodata (float, optional): Value to set as nodata in the output raster.
    """
    # If output_path is a directory, append 'merged.tif'
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "merged.tif")

    src_files_to_mosaic = [rasterio.open(path) for path in input_paths]

    # Merge rasters
    mosaic, out_trans = merge(src_files_to_mosaic, nodata=nodata)

    # Get metadata from first raster
    first_src = src_files_to_mosaic[0]
    band_descriptions = [first_src.descriptions[i] for i in range(first_src.count)]
    dtypes = first_src.dtypes
    # Remove alpha bands
    keep_indices = [i for i, desc in enumerate(band_descriptions) if desc and desc.lower() != 'alpha']
    mosaic = mosaic[keep_indices]
    dtype = dtypes[keep_indices[0]]  # assumes consistent dtype

    # Update metadata
    out_meta = first_src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "count": len(keep_indices),
        "compress": "lzw",
        "BIGTIFF": "YES",
        "dtype": dtype
    })

    if nodata is not None:
        out_meta["nodata"] = nodata

    # Write output
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)
        for i, idx in enumerate(keep_indices):
            dest.set_band_description(i + 1, band_descriptions[idx])

    # Close all sources
    for src in src_files_to_mosaic:
        src.close()


def clip_raster_to_geom(raster_path, geometry_input, output_path, nodata=None):
    """
    Clips a raster to the given geometry or bounds and retains metadata including dtype and band descriptions.
    
    Parameters:
    - raster_path (str): Path to the input raster file.
    - geometry_input (GeoSeries or bounds): gdf.geometry or gdf.bounds.
    - output_path (str): Path to save the clipped raster.
    - nodata (float, optional): Value to set as nodata in the output raster.
    """

    # Handle bounds or geometry
    if hasattr(geometry_input, "geometry"):  # likely a GeoDataFrame
        geometries = geometry_input.geometry
    elif isinstance(geometry_input, (pd.DataFrame, pd.Series)):
        bounds = geometry_input.iloc[0] if isinstance(geometry_input, pd.DataFrame) else geometry_input
        geometries = [box(bounds["minx"], bounds["miny"], bounds["maxx"], bounds["maxy"])]
    else:
        raise ValueError("geometry_input must be either gdf.geometry or gdf.bounds")

    geojson_geoms = [geom.__geo_interface__ for geom in geometries]

    # If output_path is a directory, use input filename
    if os.path.isdir(output_path):
        
        # Use the filename of the first input raster
        filename = os.path.basename(raster_path)
        output_path = os.path.join(output_path, filename)
        print(output_path)

    with rasterio.open(raster_path) as src:
        # Read and clip
        out_image, out_transform = mask(src, geojson_geoms, crop=True, nodata=nodata, filled=True)
        out_meta = src.meta.copy()

        # Remove alpha band if present
        band_descriptions = [src.descriptions[i] for i in range(src.count)]
        keep_indices = [i for i, desc in enumerate(band_descriptions) if desc and desc.lower() != 'alpha']
        out_image = out_image[keep_indices]
        print(band_descriptions)
        print(keep_indices)
        # Preserve dtype
        dtype = src.dtypes[keep_indices[0]]  # assume all kept bands have same dtype

        # Update metadata
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "count": len(keep_indices),
            "compress": "lzw",
            #"crs": "EPSG:2193",
            "dtype": dtype
        })

        if nodata is not None:
            out_meta["nodata"] = nodata

        # Write output
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)
            for i, idx in enumerate(keep_indices):
                dest.set_band_description(i + 1, band_descriptions[idx])


import rasterio
import numpy as np
import os

def raster_convert(
    input_path,
    output_dir,
    separate=None,
    current_nodata=None,
    new_nodata=None,
    actual_min = None,
    actual_max = None,
    actual_dtype=None,
    out_min = None,
    out_max = None,
    out_dtype=None
):
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(input_path) as src:
        input_dtype = src.dtypes[0]
        bands = list(range(1, src.count + 1)) if separate is None else [b for band_list in separate.values() for b in band_list]

        # Determine nodata
        src_nodata = src.nodata if current_nodata is None else current_nodata
        tgt_nodata = src_nodata if new_nodata is None else new_nodata

        # Determine dtype
        assumed_dtype = actual_dtype if actual_dtype else input_dtype
        target_dtype = out_dtype if out_dtype else assumed_dtype

        # Get target range
        if np.issubdtype(np.dtype(target_dtype), np.integer):
            target_min = 0
            target_max = np.iinfo(target_dtype).max
        else:
            target_min = 0.0
            target_max = 1.0

        # Prepare base profile
        base_profile = src.profile.copy()
        base_profile.update({
            'compress': 'LZW',
            'crs': src.crs,
            'dtype': target_dtype
        })
        if tgt_nodata is not None:
            base_profile.update({'nodata': tgt_nodata})

        # Define band groups
        band_groups = separate if separate else {f'band_{i}': [i] for i in bands}

        for name, band_indices in band_groups.items():
            data = src.read(band_indices).astype(np.float32)

            # Apply nodata mask
            if src_nodata is not None:
                mask = src.read(band_indices) == src_nodata
            else:
                mask = np.zeros_like(data, dtype=bool)

            # Determine actual value range
            valid_data = data[~mask]
            actual_min = valid_data.min() if valid_data.size > 0 else 0
            actual_max = valid_data.max() if valid_data.size > 0 else 1

            # Stretch values
            scale = (target_max - target_min) / (actual_max - actual_min) if actual_max != actual_min else 1
            stretched = (data - actual_min) * scale + target_min
            stretched = np.clip(stretched, target_min, target_max)

            # Apply nodata
            stretched[mask] = tgt_nodata if tgt_nodata is not None else 0

            # Convert to target dtype
            stretched = stretched.astype(target_dtype)

            profile = base_profile.copy()
            profile.update({'count': len(band_indices)})

            with rasterio.open(os.path.join(output_dir, f"{name}.tif"), 'w', **profile) as dst:
                dst.write(stretched)



def calculate_vegetation_indices(r_path, g_path, nir_path, re_path, output_dir):
    def read_band(path):
        with rasterio.open(path) as src:
            band = src.read(1).astype('float32')
            band[band == 65535] = np.nan
            return band, src.profile

    def normalize_clip_to_minus1_to_1(data, clip_min=0, clip_max=65535):
        clipped = np.clip(data, clip_min, clip_max)
        normalized = (clipped / ((clip_max - clip_min) / 2)) - 1
        return normalized


    def write_index(data, name):
        with rasterio.open(f"{output_dir}/{name}.tif", 'w', **profile) as dst:
            dst.write(np.nan_to_num(data, nan=-9999).astype('float32'), 1)

    # Read bands
    r, profile = read_band(r_path)
    g, _ = read_band(g_path)
    nir, _ = read_band(nir_path)
    re, _ = read_band(re_path)

    # Update profile
    profile.update({
        'count': 1,
        'compress': 'LZW',
        'crs': 'EPSG:2193',
        'dtype': 'float32',
        'nodata': -9999
    })

    epsilon = 1e-6

    # Vegetation Indices
    ndvi = (nir - r) / (nir + r + epsilon)
    ngrdi = (g - r) / (g + r + epsilon)
    re_ndvi = (nir - re) / (nir + re + epsilon)
    redvi = (re - r) / (re + r + epsilon)
    # mcari_raw = ((re - r) - 0.2 * (re - g)) * (re / (r + epsilon))
    # ari_raw = 1/(g+epsilon) - 1/(re + epsilon)

    # Clip and normalize MCARI
    # mcari = normalize_clip_to_minus1_to_1(mcari_raw)
    # ari = normalize_clip_to_minus1_to_1(ari_raw, 0, 0.0002)

    # Write outputs
    write_index(ndvi, 'NDVI')
    write_index(ngrdi, 'NGRDI')
    write_index(re_ndvi, 'RENDVI')
    write_index(redvi, 'REDVI')
    # write_index(mcari_raw, 'MCARI')
    # write_index(ari_raw, 'ARI')


