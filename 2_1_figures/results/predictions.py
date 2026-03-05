import sys
import os

ONEDRIVE = "/Users/robinpfaff/Library/CloudStorage/OneDrive-AUTUniversity/MA/aa566b206b36b985ac2ad0e73eedfc197cc8d2ffc"

# Add the modules directory to the Python path
modules_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../4_1_modules'))
sys.path.insert(0, modules_path)

import rasterio
from rasterio.windows import Window
from rasterio.mask import mask
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from shapely.geometry import box
import geopandas as gpd
import os
import pandas as pd
from config.paths import use

def get_model_prediction_path(reserve_code, bands_type, lr, loss_type, weight, layer='unet_test_zone', model_type='best_maire_f1', out_dir=None, multisite_reserves=None):
    """
    Construct the path to model predictions based on parameters.
    
    Args:
        reserve_code: Reserve code for prediction (e.g., 'ESK')
        bands_type: Band combination (e.g., 'ms_rel', 'rgb')
        lr: Learning rate (e.g., 0.02, 5e-5)
        loss_type: Loss function (e.g., 'bce_dice', 'dice')
        weight: Class weight (e.g., 1, 10, 50)
        layer: Test zone layer name
        model_type: Type of model checkpoint to use
        out_dir: Base output directory (if None, uses default relative to this script)
        multisite_reserves: List of reserves for multisite model (e.g., ['KAU', 'ESK', 'BUS', 'HAM'])
                           If None, assumes single-site model
    
    Returns:
        Path to prediction raster
    """
    # If out_dir not provided, use default relative to this script's location
    if out_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(ONEDRIVE, '5_3_unet')
    
    # Format components
    bands_upper = bands_type.upper()
    loss_upper = loss_type.upper().replace('_', '')  # e.g., 'bce_dice' -> 'BCEDICE'
    
    # Format learning rate: 0.02 -> 0_02, 5e-5 -> 5E_05
    if lr >= 0.01:
        lr_str = f"{lr:.2f}".replace('.', '_')  # e.g., 0.02 -> 0_02
    else:
        lr_str = f"{lr:.0e}".replace('e-0', 'E_0').replace('e-', 'E_')  # e.g., 5e-05 -> 5E_05
    
    # Build paths following pipeline structure
    if multisite_reserves:
        # Multisite: RGB_KAU_ESK_BUS_HAM/MW10_LR0_02_BCEDICE/test_predictions/best_maire_f1_ESK_unet_test_zone.tif
        dataset_name = f"{bands_upper}_{'_'.join(multisite_reserves)}"
        filename = f'{model_type}_{reserve_code}_{layer}.tif'
    else:
        # Single site: RGB_ESK/MW10_LR0_02_BCEDICE/test_predictions/best_maire_f1_unet_test_zone.tif
        dataset_name = f"{bands_upper}_{reserve_code}"
        filename = f'{model_type}_{layer}.tif'
    
    model_name = f"MW{weight}_LR{lr_str}_{loss_upper}"
    
    # Use absolute path to 5_3_unet folder
    base_dir = os.path.abspath(out_dir)
    
    # Prediction path
    pred_path = os.path.join(
        base_dir,
        'models',
        dataset_name,
        model_name,
        'test_predictions',
        filename
    )
    
    return pred_path

def get_f1_score(reserve_code, bands_type, lr, loss_type, weight, multisite_reserves=None, out_dir=None):
    """
    Get F1 score for a model from test_metrics.csv in the model folder.
    
    Args:
        reserve_code: Reserve code (e.g., 'ESK')
        bands_type: Band combination (e.g., 'ms_rel', 'rgb')
        lr: Learning rate (e.g., 0.02, 5e-5)
        loss_type: Loss function (e.g., 'bce_dice', 'dice')
        weight: Class weight (e.g., 1, 10, 50)
        multisite_reserves: List of reserves for multisite model
        out_dir: Base output directory
    
    Returns:
        F1 score as float, or None if not found
    """
    if out_dir is None:
        out_dir = os.path.join(ONEDRIVE, '5_3_unet')
    
    # Build path to test_metrics.csv using same logic as get_model_prediction_path
    bands_upper = bands_type.upper()
    loss_upper = loss_type.upper().replace('_', '')
    
    if lr >= 0.01:
        lr_str = f"{lr:.2f}".replace('.', '_')
    else:
        lr_str = f"{lr:.0e}".replace('e-0', 'E_0').replace('e-', 'E_')
    
    if multisite_reserves:
        dataset_name = f"{bands_upper}_{'_'.join(multisite_reserves)}"
    else:
        dataset_name = f"{bands_upper}_{reserve_code}"
    
    model_name = f"MW{weight}_LR{lr_str}_{loss_upper}"
    
    # For multisite: best_maire_f1_ESK_unet_test_zone_metrics.csv
    # For single-site: best_maire_f1_unet_test_zone_metrics.csv
    if multisite_reserves:
        metrics_filename = f'best_maire_f1_{reserve_code}_unet_test_zone_metrics.csv'
    else:
        metrics_filename = 'best_maire_f1_unet_test_zone_metrics.csv'
    
    metrics_path = os.path.join(
        out_dir,
        'models',
        dataset_name,
        model_name,
        'test_predictions',
        metrics_filename
    )
    
    try:
        if os.path.exists(metrics_path):
            df_metrics = pd.read_csv(metrics_path)
            
            # Both single-site and multisite have 'Maire F1' column
            if 'Maire F1' in df_metrics.columns:
                return df_metrics['Maire F1'].values[0]
    except Exception as e:
        print(f"Error reading {metrics_path}: {e}")
    
    return None

def get_center_bbox(test_zone_gpkg, layer='unet_test_zone', extent=15, offset_x=0, offset_y=0):
    """
    Get a square bounding box from the center of the test zone.
    
    Args:
        test_zone_gpkg: Path to test zone geopackage
        layer: Layer name in geopackage
        extent: Size of the square in meters (default: 10)
        offset_x: Horizontal offset in meters (positive = east, negative = west)
        offset_y: Vertical offset in meters (positive = north, negative = south)
    
    Returns:
        shapely box geometry
    """
    gdf = gpd.read_file(test_zone_gpkg, layer=layer)
    
    # Get centroid of test zone and apply offset
    centroid = gdf.union_all().centroid
    center_x = centroid.x + offset_x
    center_y = centroid.y + offset_y
    
    # Create square box around adjusted center
    half_extent = extent / 2
    bbox = box(
        center_x - half_extent,
        center_y - half_extent,
        center_x + half_extent,
        center_y + half_extent
    )
    
    return bbox

def plot_model_comparisons(reserve_codes, models, layer='unet_test_zone', figsize=(13, 25), save_path=None, offset_x=0, offset_y=0):
    """
    Plot RGB, labels, and predictions from multiple models for comparison across reserves.
    
    Args:
        reserve_codes: List of reserve codes (e.g., ['ESK', 'KAU', 'BUS', 'HAM'])
        models: List of model config dicts, each containing:
                {'band_comb': 'ms_rel', 'lr': 0.02, 'loss': 'bce_dice', 'weight': 10, 'label': 'Model 1'}
                For multisite models, add: 'multisite': ['KAU', 'ESK', 'BUS', 'HAM']
        layer: Test zone layer name
        figsize: Figure size tuple
        save_path: Optional path to save figure
        offset_x: Horizontal offset in meters from test zone center (positive = east, negative = west)
                  Can be a single value (applied to all reserves) or a list matching reserve_codes length
        offset_y: Vertical offset in meters from test zone center (positive = north, negative = south)
                  Can be a single value (applied to all reserves) or a list matching reserve_codes length
    
    Example:
        models = [
            {'band_comb': 'ms_rel', 'lr': 0.02, 'loss': 'bce_dice', 'weight': 10, 'label': 'MS_REL W10'},
            {'band_comb': 'rgb', 'lr': 0.02, 'loss': 'bce_dice', 'weight': 10, 'label': 'RGB W10'},
            {'band_comb': 'rgb', 'lr': 0.02, 'loss': 'bce_dice', 'weight': 10, 'label': 'RGB Multisite', 
             'multisite': ['KAU', 'ESK', 'BUS', 'HAM']},
        ]
        # Different offset for each reserve
        plot_model_comparisons(['ESK', 'KAU', 'BUS', 'HAM'], models, offset_x=[0,0,0,5], offset_y=[0,0,0,5])
    """
    # Map reserve codes to site codes
    reserve_to_site = {
        'ESK': 'A1',
        'KAU': 'A2',
        'BUS': 'A3',
        'HAM': 'H1'
    }
    
    # Setup figure: (2 + N models) rows × (1 + len(reserves)) columns (extra column for labels)
    n_rows = 2 + len(models)  # RGB + Labels + Models
    n_cols = len(reserve_codes)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Ensure axes is 2D
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Convert offset_x and offset_y to lists if they're single values
    if not isinstance(offset_x, (list, tuple)):
        offset_x = [offset_x] * len(reserve_codes)
    if not isinstance(offset_y, (list, tuple)):
        offset_y = [offset_y] * len(reserve_codes)
    
    # Add row labels on the left
    row_labels = ['RGB', 'Ground Truth'] + [model.get('label', f"Model {i+1}") for i, model in enumerate(models)]
    
    # Process each reserve (column)
    for col_idx, reserve_code in enumerate(reserve_codes):
        r = use([reserve_code])[0]
        
        # Get 10x10m bbox from test zone center with reserve-specific offset
        test_zone_gpkg = os.path.join(r.GIS, 'bbox.gpkg')
        bbox_10m = get_center_bbox(test_zone_gpkg, layer, offset_x=offset_x[col_idx], offset_y=offset_y[col_idx])
        bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox_10m]}, crs='EPSG:2193')
        
        # Read the training zone for masking
        training_zone_gdf = gpd.read_file(test_zone_gpkg, layer=layer)
        
        # First, determine the reference dimensions from RGB to ensure consistency
        with rasterio.open(r.P_RGB_015) as src:
            out_image, out_transform = mask(src, bbox_gdf.geometry, crop=True)
            reference_height, reference_width = out_image.shape[1:]
            rgb = np.moveaxis(out_image[:3], 0, -1)
            rgb = rgb / rgb.max()
        
        # Create training zone mask
        from rasterio.features import rasterize
        training_mask = rasterize(
            [(geom, 1) for geom in training_zone_gdf.geometry],
            out_shape=(reference_height, reference_width),
            transform=out_transform,
            fill=0,
            dtype=np.uint8
        ).astype(bool)
        
        # Row 0: Plot RGB with training zone mask
        ax_rgb = axes[0, col_idx]
        # Create grey background
        rgb_masked = np.ones((reference_height, reference_width, 3)) * 0.85  # Light grey
        # Apply RGB only within training zone
        rgb_masked[training_mask] = rgb[training_mask]
        ax_rgb.imshow(rgb_masked, extent=[0, reference_width, reference_height, 0])
        ax_rgb.set_xlim(0, reference_width)
        ax_rgb.set_ylim(reference_height, 0)
        # Use site code (A1, A2, etc.) for title instead of reserve code
        site_code = reserve_to_site.get(reserve_code, reserve_code)
        ax_rgb.set_title(f'{site_code}', fontsize=14)
        ax_rgb.axis('off')
        
        # Add row label on the leftmost column
        if col_idx == 0:
            ax_rgb.text(-0.03, 0.5, row_labels[0], transform=ax_rgb.transAxes,
                      fontsize=14, rotation=90,
                      verticalalignment='center', horizontalalignment='right')
        
        # Row 1: Plot Labels
        ax_label = axes[1, col_idx]
        label_path = os.path.join(r.GIS, 'swamp_maire_poly.gpkg')
        if os.path.exists(label_path):
            label_gdf = gpd.read_file(label_path, layer='maire_poly_ms')
            
            label_mask = rasterize(
                [(geom, 1) for geom in label_gdf.geometry],
                out_shape=(reference_height, reference_width),
                transform=out_transform,
                fill=0,
                dtype=np.uint8
            )
            
            # Apply training zone mask - 0.85 for light grey outside training zone
            label_display = np.where(training_mask, label_mask, 0.85)
            
            ax_label.imshow(label_display, cmap='gray', vmin=0, vmax=1, extent=[0, reference_width, reference_height, 0])
            ax_label.set_xlim(0, reference_width)
            ax_label.set_ylim(reference_height, 0)
            ax_label.axis('off')
            
            # Add row label on the leftmost column
            if col_idx == 0:
                ax_label.text(-0.03, 0.5, row_labels[1], transform=ax_label.transAxes,
                            fontsize=14, rotation=90,
                            verticalalignment='center', horizontalalignment='right')
        else:
            ax_label.text(0.5, 0.5, 'Labels not found', ha='center', va='center', fontsize=8)
            ax_label.axis('off')
            
            if col_idx == 0:
                ax_label.text(-0.03, 0.5, row_labels[1], transform=ax_label.transAxes,
                            fontsize=14, rotation=90,
                            verticalalignment='center', horizontalalignment='right')
        
        # Rows 2+: Plot Model Predictions
        for model_idx, model_config in enumerate(models):
            ax = axes[2 + model_idx, col_idx]
            
            pred_path = get_model_prediction_path(
                reserve_code,
                model_config['band_comb'],
                model_config['lr'],
                model_config['loss'],
                model_config['weight'],
                layer=layer,
                multisite_reserves=model_config.get('multisite')
            )
            
            if not os.path.exists(pred_path):
                ax.text(0.5, 0.5, f'Not found', ha='center', va='center', fontsize=8)
                ax.axis('off')
                
                # Add row label on the leftmost column
                if col_idx == 0:
                    ax.text(-0.03, 0.5, row_labels[2 + model_idx], transform=ax.transAxes,
                          fontsize=14, rotation=90,
                          verticalalignment='center', horizontalalignment='right')
                continue
            
            with rasterio.open(pred_path) as src:
                try:
                    # Read prediction with the same geographic extent as RGB
                    from rasterio.warp import reproject, Resampling
                    
                    # Create output array with reference dimensions
                    padded_pred = np.zeros((reference_height, reference_width), dtype=np.uint8)
                    
                    # Reproject prediction to match RGB resolution and extent
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=padded_pred,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=out_transform,
                        dst_crs='EPSG:2193',
                        resampling=Resampling.nearest
                    )
                    
                    pred_binary = (padded_pred > 0).astype(np.uint8)
                    
                    # Apply training zone mask - 0.85 for light grey outside training zone
                    pred_display = np.where(training_mask, pred_binary, 0.85)
                    
                    # Create colored prediction display
                    # Check if this is an RGB or MS model based on band_comb
                    band_comb = model_config['band_comb']
                    if band_comb == 'rgb':
                        # Yellow for RGB predictions: [R, G, B] = [0.95, 0.85, 0.0]
                        pred_colored = np.stack([
                            np.where(pred_display == 1, 0.95, pred_display),  # R channel
                            np.where(pred_display == 1, 0.85, pred_display),  # G channel
                            np.where(pred_display == 1, 0.0, pred_display)   # B channel
                        ], axis=-1)
                    elif 'ms' in band_comb or 'ind' in band_comb:
                        # Purple for MS predictions: [R, G, B] = [0.7, 0.3, 0.8]
                        pred_colored = np.stack([
                            np.where(pred_display == 1, 0.7, pred_display),  # R channel
                            np.where(pred_display == 1, 0.3, pred_display),  # G channel
                            np.where(pred_display == 1, 0.8, pred_display)   # B channel
                        ], axis=-1)
                    else:
                        # Default grayscale
                        pred_colored = np.stack([pred_display] * 3, axis=-1)
                    
                    ax.imshow(pred_colored, extent=[0, reference_width, reference_height, 0])
                    ax.set_xlim(0, reference_width)
                    ax.set_ylim(reference_height, 0)
                    ax.axis('off')
                    
                    # Get and display F1 score
                    f1_score = get_f1_score(
                        reserve_code,
                        model_config['band_comb'],
                        model_config['lr'],
                        model_config['loss'],
                        model_config['weight'],
                        multisite_reserves=model_config.get('multisite')
                    )
                    
                    if f1_score is not None:
                        # Add F1 score in bottom right corner with white background
                        ax.text(0.98, 0.98, f'F1: {f1_score:.2f}',
                               transform=ax.transAxes,
                               fontsize=11,
                               verticalalignment='top',
                               horizontalalignment='right',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
                    
                except Exception as e:
                    # If bbox doesn't intersect with prediction, show light grey image
                    ax.imshow(np.ones((reference_height, reference_width)) * 0.85, 
                             cmap='gray', vmin=0, vmax=1, extent=[0, reference_width, reference_height, 0])
                    ax.set_xlim(0, reference_width)
                    ax.set_ylim(reference_height, 0)
                    ax.axis('off')
                    print(f"Error processing {pred_path}: {e}")
                
                # Add row label on the leftmost column
                if col_idx == 0:
                    ax.text(-0.03, 0.5, row_labels[2 + model_idx], transform=ax.transAxes,
                          fontsize=14, rotation=90,
                          verticalalignment='center', horizontalalignment='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    plt.show()
