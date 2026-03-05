"""
Path builder module for field site data directories.
Constructs standardized path structures for raw data, processed outputs, and datasets.
"""

import os
from types import SimpleNamespace
from .config import DATA

def site_paths(extent: str, folder: str, prefix: str = ''):
    """
    Generate standardized path dictionary for a field site.
    
    Args:
        extent: Spatial extent folder name (e.g., 'limited_extent', 'full_extent')
        folder: Site-specific folder name
        prefix: Optional prefix for path keys
        
    Returns:
        Dictionary mapping path keys to full directory paths
    """
    base = os.path.join(DATA, extent, folder)
    return {
        f'{prefix}BASE': base,
        f'{prefix}R': os.path.join(base, 'A_raw'),
        f'{prefix}P': os.path.join(base, 'B_processed'),
        f'{prefix}GPS': os.path.join(base, 'C_gps'),
        f'{prefix}EXT': os.path.join(base, 'D_external'),
        f'{prefix}GIS': os.path.join(base, 'E_gis'),
        f'{prefix}DS': os.path.join(base, 'F_datasets'),
        f'{prefix}OUT': os.path.join(base, 'G_outputs'),
        f'{prefix}RES': os.path.join(base, 'H_results'),
        f'{prefix}I': os.path.join(base, 'I_photos_videos'),
        f'{prefix}R_LAS': os.path.join(base, 'A_raw/lidar'),
        f'{prefix}R_MS': os.path.join(base, 'A_raw/ms'),
        f'{prefix}R_RGB': os.path.join(base, 'A_raw/rgb'),
        f'{prefix}P_LAS': os.path.join(base, 'B_processed/lidar'),
        f'{prefix}P_MS': os.path.join(base, 'B_processed/ms'),
        f'{prefix}P_MS_025': os.path.join(base, 'B_processed/ms/ms_025.tif'),
        f'{prefix}P_RGB': os.path.join(base, 'B_processed/rgb'),
        f'{prefix}P_RGB_015': os.path.join(base, 'B_processed/rgb/rgb_015.tif'),
        f'{prefix}P_RGB_R': os.path.join(base, 'B_processed/rgb/r.tif'),
        f'{prefix}P_RGB_G': os.path.join(base, 'B_processed/rgb/g.tif'),
        f'{prefix}P_RGB_B': os.path.join(base, 'B_processed/rgb/b.tif'),
        f'{prefix}P_IND': os.path.join(base, 'B_processed/indices'),
        f'{prefix}P_MS_R': os.path.join(base, 'B_processed/ms/r_uint8.tif'),
        f'{prefix}P_MS_G': os.path.join(base, 'B_processed/ms/g_uint8.tif'),
        f'{prefix}P_MS_RE': os.path.join(base, 'B_processed/ms/re_uint8.tif'),
        f'{prefix}P_MS_NIR': os.path.join(base, 'B_processed/ms/nir_uint8.tif'),
        f'{prefix}P_IND_NDVI': os.path.join(base, 'B_processed/indices/NDVI_uint8.tif'),
        f'{prefix}P_IND_ARI': os.path.join(base, 'B_processed/indices/ARI_uint8.tif'),
        f'{prefix}P_IND_NGRDI': os.path.join(base, 'B_processed/indices/NGRDI_uint8.tif'),
        f'{prefix}P_IND_RENDVI': os.path.join(base, 'B_processed/indices/RENDVI_uint8.tif'),
        f'{prefix}P_IND_REDVI': os.path.join(base, 'B_processed/indices/REDVI_uint8.tif'),
        f'{prefix}P_IND_MCARI': os.path.join(base, 'B_processed/indices/MCARI_uint8.tif'),
        f'{prefix}P_DTM': os.path.join(base, 'B_processed/DTM'),
        f'{prefix}P_DSM': os.path.join(base, 'B_processed/DSM'),
        f'{prefix}DS_UNET_OLD': os.path.join(base, 'F_datasets/unet_old'), 
        f'{prefix}DS_UNET': os.path.join(base, 'F_datasets/unet'), 
        f'{prefix}OUT_LAS': os.path.join(base, 'G_outputs/lidar'),
        f'{prefix}OUT_MS': os.path.join(base, 'G_outputs/ms'),
        f'{prefix}RES_LAS': os.path.join(base, 'H_results/lidar'),
        f'{prefix}RES_MS': os.path.join(base, 'H_results/ms'),
    }

def build_site(extent_folder: str, folder: str, key: str = None):
    """
    Build a site object with both full extent and limited extent paths.
    
    Args:
        extent_folder: Limited extent folder name
        folder: Site-specific folder name
        key: Optional site identifier key
        
    Returns:
        SimpleNamespace object with path attributes accessible via dot notation
    """
    paths = {}
    paths.update(site_paths('full_extent', folder, prefix='FE_'))
    paths.update(site_paths(extent_folder, folder))  # default = no prefix
    if key:
        paths['name'] = key
    return SimpleNamespace(**paths)


