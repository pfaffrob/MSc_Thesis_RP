"""
U-Net preprocessing utilities for preparing training data.
Handles vector rasterization, image tiling, and dataset distribution.
"""

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.mask import mask
from rasterio.windows import Window
import numpy as np
import os
import shutil
import glob
import random
from tqdm import tqdm
from shapely.geometry import box

def rasterize_vector(poly_gdf, raster_path, output_path, tile_size=8192, mask_gpkg = None, mask_layer  = None):
    """
    This function rasterizes a vector layer from a GeoPackage file and saves the result to a raster file in tiles.

    Parameters:
    vector_path (str): The path to the input GeoPackage file.
    layer_name (str): The name of the vector layer within the GeoPackage.
    raster_path (str): The path to the source raster file used for getting the metadata and the raster extent.
    output_path (str): The path where the output raster file will be saved.
    tile_size (int): The size of the tiles to process.

    Returns:
    None
    """
    # Read vector data from GeoPackage
    gdf = poly_gdf

    if mask_gpkg and mask_layer:
        mask_gdf = gpd.read_file(mask_gpkg, layer=mask_layer)
        gdf = gdf.clip(mask_gdf)

    # Read source raster
    with rasterio.open(raster_path) as src:

        gdf['geometry'] = gdf['geometry'].apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)

        # Check for empty or invalid geometries
        if gdf.is_empty.any():
            print("Warning: Empty geometries found.")
        if not gdf.is_valid.all():
            print("Warning: Invalid geometries found.")

        raster_extent = src.bounds
        vector_extent = gdf.total_bounds

        # Create bounding boxes
        raster_bbox = box(*raster_extent)
        vector_bbox = box(*vector_extent)

        # Check if the bounding boxes intersect
        if not raster_bbox.intersects(vector_bbox):
            print("Warning: Vector extent does not overlap with raster extent.")

        # Prepare metadata for output raster
        meta = src.profile.copy()
        meta.update({
            'count': 1,
            'compress': 'lzw',
            'dtype': 'uint8',  # Ensure the data type is uint8
            'nodata': 255
        })

        # Create the output raster file
        with rasterio.open(output_path, 'w', **meta) as dst:
            # Calculate the number of tiles in each dimension
            num_tiles_x = (src.width + tile_size - 1) // tile_size
            num_tiles_y = (src.height + tile_size - 1) // tile_size
            total_tiles = num_tiles_x * num_tiles_y

            # Process in tiles with a progress bar
            with tqdm(total=total_tiles, desc="Rasterizing", unit="tiles") as pbar:
                for i in range(num_tiles_y):
                    for j in range(num_tiles_x):
                        # Define the window
                        window = Window(
                            col_off=j * tile_size,
                            row_off=i * tile_size,
                            width=min(tile_size, src.width - j * tile_size),
                            height=min(tile_size, src.height - i * tile_size)
                        )

                        # Get the bounds of the current window
                        window_bounds = rasterio.windows.bounds(window, src.transform)
                        window_geom = box(*window_bounds)

                        # Read the window from the source raster
                        src_window = src.read(window=window, indexes=1)

                        # Rasterize the vector data within the window
                        shapes = [(geom, 1) for geom in gdf.geometry if geom.intersects(window_geom)]
                        if shapes:  # Check if there are valid shapes
                            rasterized = rasterize(
                                shapes,
                                out_shape=(window.height, window.width),
                                transform=src.window_transform(window),
                                fill=0,
                                default_value=1,
                                dtype='uint8'
                            )

                            # Write the rasterized data to the output file
                            dst.write(rasterized, window=window, indexes=1)

                        # Update the progress bar
                        pbar.update(1)

    print(f"Rasterization complete. Output saved to {output_path}")

import os
import shutil
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.windows import Window
from tqdm import tqdm

def tile_image(gdf_path, tif_paths, output_dir, file_prefix='', layer=None, tile_size=572, clear_dir=True):
    os.makedirs(output_dir, exist_ok=True)
    if clear_dir:
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    # List to store the paths of the intermediate files
    intermediate_files = []
    # Load the polygons
    if layer is not None:
        gdf = gpd.read_file(gdf_path, layer=layer)
    else:
        gdf = gpd.read_file(gdf_path)

    if isinstance(tif_paths, str):
        tif_paths = [tif_paths]

    raster_infos = []
    for path in tif_paths:
        with rasterio.open(path) as src:
            if src.count != 1:
                raise ValueError(f"{path} is not a single-band raster.")
            raster_infos.append((src.width, src.height, src.crs, src.transform))

    ref_width, ref_height, ref_crs, ref_transform = raster_infos[0]
    for i, (w, h, crs, transform) in enumerate(raster_infos[1:], start=1):
        if (w != ref_width or h != ref_height or crs != ref_crs or transform != ref_transform):
            print(ref_width, ref_height, ref_crs, ref_transform)
            print(w, h, crs, transform)
            raise ValueError(f"Raster at index {i} does not match resolution, CRS, or transform.")

    with rasterio.open(tif_paths[0]) as ref_src:
        meta = ref_src.meta.copy()

        for i, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
            clipped_bands = []
            for tif_path in tif_paths:
                with rasterio.open(tif_path) as src:
                    clipped_image, clipped_transform = mask(src, [row.geometry], crop=True, nodata=0)
                    clipped_bands.append(clipped_image[0])

            stacked_image = np.stack(clipped_bands)
            clipped_meta = meta.copy()
            clipped_meta.update({
                "driver": "GTiff",
                "height": stacked_image.shape[1],
                "width": stacked_image.shape[2],
                "transform": clipped_transform,
                "count": len(tif_paths),
                "nodata": 0
            })

            clipped_output_path = os.path.join(output_dir, f'masked_{i}.tif')
            with rasterio.open(clipped_output_path, 'w', **clipped_meta, compress='lzw') as dest:
                dest.write(stacked_image)

            intermediate_files.append(clipped_output_path)

            overlap = 0.25
            step = int(tile_size * (1 - overlap))

            pad_height = max(tile_size - stacked_image.shape[1], 0)
            pad_width = max(tile_size - stacked_image.shape[2], 0)

            if pad_height > 0 or pad_width > 0:
                padded_image = np.pad(stacked_image,
                                      ((0, 0), (0, pad_height), (0, pad_width)),
                                      mode='constant', constant_values=0)
                padded_meta = clipped_meta.copy()
                padded_meta.update({
                    "height": padded_image.shape[1],
                    "width": padded_image.shape[2]
                })

                padded_output_path = os.path.join(output_dir, f'padded_masked_{i}.tif')
                with rasterio.open(padded_output_path, 'w', **padded_meta, compress='lzw') as dest:
                    dest.write(padded_image)
                intermediate_files.append(padded_output_path)
            else:
                padded_image = stacked_image
                padded_meta = clipped_meta
                padded_output_path = clipped_output_path

            with rasterio.open(padded_output_path) as padded_src:
                for row_start in range(0, padded_src.height, step):
                    for col_start in range(0, padded_src.width, step):
                        if row_start + tile_size > padded_src.height or col_start + tile_size > padded_src.width:
                            continue  # Skip tiles that would exceed bounds
                        window = Window(col_start, row_start, tile_size, tile_size)
                        tile = padded_src.read(window=window)

                        if np.count_nonzero(tile) == 0:
                            continue

                        tile_meta = padded_src.meta.copy()
                        tile_meta.update({
                            "driver": "GTiff",
                            "height": tile_size,
                            "width": tile_size,
                            "transform": rasterio.windows.transform(window, padded_src.transform),
                            "nodata": 0
                        })

                        tile_output_path = os.path.join(output_dir, f'{file_prefix}{i}_{row_start}_{col_start}.tif')
                        with rasterio.open(tile_output_path, 'w', **tile_meta, compress='lzw') as dest:
                            dest.write(tile)


    for file_path in intermediate_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            


def clip_geotiffs(input_tiff, reference_dir, output_dir):
    """
    Clips an input GeoTIFF to the bounds of reference GeoTIFFs in a specified directory and saves the results.

    Parameters:
    input_tiff (str): The path to the input GeoTIFF file to be clipped.
    reference_dir (str): The directory containing reference GeoTIFF files used for clipping.
    output_dir (str): The directory where the clipped GeoTIFF files will be saved.

    Returns:
    None
    """
    # Get a list of all reference GeoTIFFs in the directory
    reference_tiffs = glob.glob(os.path.join(reference_dir, '*.tif'))
    os.makedirs(output_dir, exist_ok=True)
    # Initialize the progress bar
    with tqdm(total=len(reference_tiffs), desc="Clipping GeoTIFFs", unit="file") as pbar:
        for reference_tiff in reference_tiffs:
            try:
                # Open the reference GeoTIFF to get its bounds and metadata
                with rasterio.open(reference_tiff) as ref:
                    bounds = ref.bounds
                    ref_transform = ref.transform
                    ref_width = ref.width
                    ref_height = ref.height

                # Convert bounds to a geometry object
                geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

                # Open the input GeoTIFF
                with rasterio.open(input_tiff) as src:
                    # Update the metadata for the output using the reference GeoTIFF's transform and dimensions
                    kwargs = src.meta.copy()
                    kwargs.update({
                        'driver': 'GTiff',
                        'height': ref_height,
                        'width': ref_width,
                        'transform': ref_transform,
                        'compress': 'LZW',
                        'nodata': 0  # Set nodatavalue to 0
                    })

                    # Clip the input GeoTIFF to the bounds of the reference GeoTIFF
                    out_image, out_transform = mask(src, [geom], crop=True, nodata=0)

                    # Ensure the output image has the same dimensions as the reference
                    out_image = out_image[:, :ref_height, :ref_width]

                    # Save the clipped GeoTIFF with the same basename as the reference GeoTIFF
                    output_path = os.path.join(output_dir, os.path.basename(reference_tiff))
                    with rasterio.open(output_path, 'w', **kwargs) as dst:
                        dst.write(out_image)

                # Update the progress bar
                pbar.update(1)
            except Exception as e:
                print(f"Error processing {reference_tiff}: {e}")            
            

def distribute_files(dataset_dir, folders=['train', 'test', 'valid'], distribution=[0.6, 0.2, 0.2], image_folder='aerial_576', label_folder='label_576', ignore_nomatch=False, out_dir=None):
    image_dir = os.path.join(dataset_dir, image_folder)
    label_dir = os.path.join(dataset_dir, label_folder)

    # Check if both folders contain the same filenames
    image_files = set(f.split('.')[0] for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and f.split('.')[0])
    label_files = set(f.split('.')[0] for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f)) and f.split('.')[0])

    if not ignore_nomatch:
        if image_files != label_files:
            raise ValueError("Image and label files do not match")

    # Determine the output directory
    if out_dir is None:
        out_dir = dataset_dir
    else:
        out_dir = os.path.abspath(out_dir)

    # Create folders if they don't exist
    for folder in folders:
        folder_masks = folder + '_masks'
        folder_images = folder + '_images'
        os.makedirs(os.path.join(out_dir, folder_masks), exist_ok=True)
        os.makedirs(os.path.join(out_dir, folder_images), exist_ok=True)

    # Get list of image files
    image_files = list(image_files)
    random.shuffle(image_files)

    # Calculate the number of files for each folder
    total_files = len(image_files)
    num_files = [int(total_files * dist) for dist in distribution]

    # Adjust the last folder to include any remaining files
    num_files[-1] += total_files - sum(num_files)

    # Distribute files
    start = 0
    for folder, count in zip(folders, num_files):
        for file_base in image_files[start:start + count]:
            # Copy image file
            image_file = file_base + '.tif'  # Adjust extension if necessary
            if os.path.exists(os.path.join(image_dir, image_file)):
                shutil.copy(os.path.join(image_dir, image_file), os.path.join(out_dir, folder + '_images', image_file))
            
            # Copy corresponding label file
            label_file = file_base + '.tif'
            if os.path.exists(os.path.join(label_dir, label_file)):
                shutil.copy(os.path.join(label_dir, label_file), os.path.join(out_dir, folder + '_masks', label_file))
        
        start += count

        import os
import shutil
import random
import rasterio
import numpy as np

def distribute_files_with_target_balance(
    dataset_dir,
    folders=['train', 'test', 'valid'],
    distribution=[0.6, 0.2, 0.2],
    image_folder='aerial_576',
    label_folder='label_576',
    ignore_nomatch=False,
    out_dir=None,
    target_threshold=1  # Minimum pixel count to consider target present
):
    """
    Distribute files to train/valid/test splits while maintaining the same
    target class balance across all splits (stratified sampling).
    
    If overall dataset is 93% background / 7% target, each split will have
    approximately the same 93/7 ratio.
    """
    image_dir = os.path.join(dataset_dir, image_folder)
    label_dir = os.path.join(dataset_dir, label_folder)

    image_files = set(f.split('.')[0] for f in os.listdir(image_dir) if f.endswith('.tif'))
    label_files = set(f.split('.')[0] for f in os.listdir(label_dir) if f.endswith('.tif'))

    if not ignore_nomatch and image_files != label_files:
        raise ValueError("Image and label files do not match")

    if out_dir is None:
        out_dir = dataset_dir
    else:
        out_dir = os.path.abspath(out_dir)

    for folder in folders:
        os.makedirs(os.path.join(out_dir, folder + '_images'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, folder + '_masks'), exist_ok=True)

    # Identify masks with target class
    target_files = []
    non_target_files = []

    for file_base in label_files:
        label_path = os.path.join(label_dir, file_base + '.tif')
        with rasterio.open(label_path) as src:
            data = src.read(1)
            if np.sum(data >= target_threshold) > 0:
                target_files.append(file_base)
            else:
                non_target_files.append(file_base)

    # Shuffle both groups independently
    random.shuffle(target_files)
    random.shuffle(non_target_files)

    # Calculate split sizes for target and non-target files separately
    # This maintains the same class balance across all splits
    num_target = len(target_files)
    num_non_target = len(non_target_files)
    
    target_splits = [int(num_target * dist) for dist in distribution]
    target_splits[-1] += num_target - sum(target_splits)  # Add remainder to last split
    
    non_target_splits = [int(num_non_target * dist) for dist in distribution]
    non_target_splits[-1] += num_non_target - sum(non_target_splits)  # Add remainder to last split

    # Distribute target files
    target_split_map = {}
    start_idx = 0
    for i, folder in enumerate(folders):
        end_idx = start_idx + target_splits[i]
        target_split_map[folder] = target_files[start_idx:end_idx]
        start_idx = end_idx

    # Distribute non-target files
    non_target_split_map = {}
    start_idx = 0
    for i, folder in enumerate(folders):
        end_idx = start_idx + non_target_splits[i]
        non_target_split_map[folder] = non_target_files[start_idx:end_idx]
        start_idx = end_idx

    # Combine and shuffle each split
    split_map = {}
    for folder in folders:
        combined = target_split_map[folder] + non_target_split_map[folder]
        random.shuffle(combined)
        split_map[folder] = combined

    # Print distribution statistics
    total_files = num_target + num_non_target
    target_ratio = num_target / total_files * 100 if total_files > 0 else 0
    print(f"\nOverall dataset balance: {num_non_target} background ({100-target_ratio:.1f}%) / {num_target} target ({target_ratio:.1f}%)")
    print(f"Distribution across splits:")
    for folder in folders:
        folder_target = len(target_split_map[folder])
        folder_total = len(split_map[folder])
        folder_ratio = folder_target / folder_total * 100 if folder_total > 0 else 0
        print(f"  {folder}: {folder_total} files ({folder_target} target = {folder_ratio:.1f}%)")

    # Copy files to output directories
    for folder in folders:
        for file_base in split_map[folder]:
            image_file = file_base + '.tif'
            label_file = file_base + '.tif'
            shutil.copy(os.path.join(image_dir, image_file), os.path.join(out_dir, folder + '_images', image_file))
            shutil.copy(os.path.join(label_dir, label_file), os.path.join(out_dir, folder + '_masks', label_file))