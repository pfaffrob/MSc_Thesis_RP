"""
U-Net semantic segmentation pipeline for multi-band imagery.
Provides end-to-end workflow from data preprocessing to model training and inference.
"""

import os
import shutil
import geopandas as gpd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from PIL import Image
from collections import defaultdict

from unet_2.src.model import UNet
from unet_2.src.engine import train_model, DiceLoss, CombinedLoss, evaluate_stitched_predictions
from unet_2.src.utils import SaveBestModel, SaveBestModelIOU, load_model
from unet_2.src.datasets import get_images, get_dataset, get_data_loaders
from unet_2.src.config import ALL_CLASSES, LABEL_COLORS_LIST, DEVICE, CLASS_WEIGHTS
from unet_2.preprocessing import rasterize_vector, tile_image, distribute_files, clip_geotiffs, distribute_files_with_target_balance

from config.paths import use
from raster.utils import get_num_bands
from utils.helper_functions import list_files, makedirs


def get_bands_for_reserve(r, bands_type, custom_bands_func=None):
    """
    Get band file paths for a reserve based on bands_type.
    
    Args:
        r: Reserve object
        bands_type: Type of bands ('ms_rel', 'rgb', 'ms_bands', 'custom', etc.)
        custom_bands_func: Function that takes 'r' and returns list of band paths (for 'custom' type)
    
    Returns:
        List of band file paths
    """
    if bands_type == 'ms_rel':
        return list_files(os.path.join(r.P_MS, 'sep_rel'), '.tif')
    elif bands_type == 'ms_abs':
        return list_files(os.path.join(r.P_MS, 'sep_abs'), '.tif')
    elif bands_type == 'ms_rel_rendvi':
        bands = list_files(os.path.join(r.P_MS, 'sep_rel'), '.tif')
        return bands + [os.path.join(r.P_IND, 'sep_rel', 'uint16', 'RENDVIuint16.tif')]
    elif bands_type == 'ms_abs_rendvi':
        bands = list_files(os.path.join(r.P_MS, 'sep_abs'), '.tif')
        return bands + [os.path.join(r.P_IND, 'sep_abs', 'uint16', 'RENDVIuint16.tif')]
    elif bands_type == 'ind_rel':
        return list_files(os.path.join(r.P_IND, 'sep_rel', 'uint16'), '.tif')
    elif bands_type == 'ind_abs':
        return list_files(os.path.join(r.P_IND, 'sep_abs', 'uint16'), '.tif')
    elif bands_type == 'rgb':
        return [r.P_RGB_R, r.P_RGB_G, r.P_RGB_B]
    elif bands_type == 'custom':
        if custom_bands_func is None:
            raise ValueError("custom_bands_func must be provided when bands_type='custom'")
        return custom_bands_func(r)
    else:
        raise ValueError(f"Invalid bands_type: {bands_type}")


def calculate_class_distribution(dataset_dir, training_dataset_dir, dataset_name):
    """
    Calculate and save class distribution statistics for the dataset.
    
    Args:
        dataset_dir: Path to dataset directory (where stats file will be saved)
        training_dataset_dir: Path to training_dataset directory (where data is located)
        dataset_name: Name of the dataset
    """
    print("Calculating class distribution...")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Training dataset directory: {training_dataset_dir}")
    
    # Count pixels for each class in train and valid splits
    splits = ['train_masks', 'valid_masks']
    stats = {}
    
    for split in splits:
        label_dir = os.path.join(training_dataset_dir, split)
        print(f"Checking {split} label directory: {label_dir}")
        
        if not os.path.exists(label_dir):
            print(f"  Directory does not exist!")
            continue
        
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.tif')]
        print(f"  Found {len(label_files)} .tif files")
        
        if len(label_files) == 0:
            print(f"  No .tif files found in {label_dir}")
            continue
            
        pixel_counts = defaultdict(int)
        total_pixels = 0
        num_images = 0
        
        # Process all label images
        for label_file in label_files:
            label_path = os.path.join(label_dir, label_file)
            
            # Read label image
            try:
                label_img = np.array(Image.open(label_path))
                num_images += 1
                
                # Count pixels for each class
                unique, counts = np.unique(label_img, return_counts=True)
                for class_id, count in zip(unique, counts):
                    pixel_counts[int(class_id)] += int(count)
                    total_pixels += int(count)
            except Exception as e:
                print(f"Warning: Could not read {label_file}: {e}")
                continue
        
        print(f"  Processed {num_images} images, {total_pixels:,} total pixels")
        print(f"  Classes found: {list(pixel_counts.keys())}")
        
        stats[split] = {
            'pixel_counts': dict(pixel_counts),
            'total_pixels': total_pixels,
            'num_images': num_images
        }
    
    if len(stats) == 0:
        print("ERROR: No statistics collected! Check that training_dataset directory contains train/valid/label subdirectories with .tif files")
        return
    
    # Write statistics to file (one level up from training_dataset_dir)
    stats_file = os.path.join(dataset_dir, 'class_distribution.txt')
    print(f"\nWriting class distribution to: {stats_file}")
    
    try:
        with open(stats_file, 'w') as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write("=" * 80 + "\n\n")
            
            for split in splits:
                if split not in stats:
                    continue
                    
                split_stats = stats[split]
                f.write(f"{split.upper()} SET:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Number of images: {split_stats['num_images']}\n")
                f.write(f"Total pixels: {split_stats['total_pixels']:,}\n\n")
                
                # Class-wise statistics
                f.write("Class Distribution:\n")
                for class_id in sorted(split_stats['pixel_counts'].keys()):
                    count = split_stats['pixel_counts'][class_id]
                    percentage = (count / split_stats['total_pixels'] * 100) if split_stats['total_pixels'] > 0 else 0
                    
                    # Get class name
                    class_name = ALL_CLASSES[class_id] if class_id < len(ALL_CLASSES) else f"Unknown_{class_id}"
                    
                    f.write(f"  Class {class_id} ({class_name:15s}): {count:12,} pixels ({percentage:6.2f}%)\n")
                
                # Calculate imbalance ratio
                if len(split_stats['pixel_counts']) == 2:
                    counts_list = sorted(split_stats['pixel_counts'].values())
                    if counts_list[0] > 0:
                        imbalance_ratio = counts_list[1] / counts_list[0]
                        f.write(f"\nClass imbalance ratio (majority/minority): {imbalance_ratio:.2f}:1\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
        
        print(f"Class distribution successfully saved to: {stats_file}")
        
    except Exception as e:
        print(f"ERROR writing file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print summary to console
    print("\nDataset Summary:")
    for split in splits:
        if split not in stats:
            continue
        split_stats = stats[split]
        print(f"\n{split.upper()}: {split_stats['num_images']} images, {split_stats['total_pixels']:,} pixels")
        for class_id in sorted(split_stats['pixel_counts'].keys()):
            count = split_stats['pixel_counts'][class_id]
            percentage = (count / split_stats['total_pixels'] * 100) if split_stats['total_pixels'] > 0 else 0
            class_name = ALL_CLASSES[class_id] if class_id < len(ALL_CLASSES) else f"Unknown_{class_id}"
            print(f"  {class_name}: {percentage:.2f}%")


def train_unet_pipeline(
    reserves,
    maire_weight,
    lr,
    bands_type='ms_rel',  # 'ms_rel', 'rgb', 'ms_bands', 'custom'
    custom_bands_func=None,  # Function that takes 'r' and returns list of band paths
    custom_bands_name=None,  # Short name for custom bands (e.g., 'G_R_RE_NIR_RENDVI')
    tile_size=576,
    batch_size=2,
    epochs=300,
    seed=42,
    out_dir='../../5_3_unet',
    label_gpkg_name='swamp_maire_poly.gpkg',
    label_layer='maire_poly_ms',
    training_zone_gpkg_name='bbox.gpkg',
    training_zone_layer='unet_training_zone',
    use_scheduler=True,
    loss_type='dice',  # 'ce', 'bce', 'dice', 'bce_dice', 'ce_dice'
    scheduler_metric='iou',  # 'loss' for LR reduction on loss plateau, 'iou' for IoU plateau
    run_inference=True  # Whether to run inference on test zones after training
):
    """
    Train UNet model with specified configuration.
    
    Args:
        reserves: List of reserve codes (e.g., ['ESK', 'KAU', 'BUS', 'HAM'])
        maire_weight: Weight for maire class in loss function
        lr: Learning rate
        bands_type: Type of bands to use ('ms_rel', 'rgb', 'ms_bands', 'custom')
        custom_bands_func: Function that takes reserve object 'r' and returns list of band paths
                          Example: lambda r: [r.P_MS_G, r.P_MS_R, r.P_MS_RE, r.P_MS_NIR, 
                                            os.path.join(r.P_IND, 'sep_rel', 'RENDVI_uint8.tif')]
        custom_bands_name: Short name for custom bands (e.g., 'G_R_RE_NIR_RENDVI')
        tile_size: Size of image tiles
        batch_size: Training batch size
        epochs: Number of training epochs
        seed: Random seed
        out_dir: Base directory for datasets and outputs
        label_gpkg_name: Name of label geopackage file
        label_layer: Layer name in label geopackage
        training_zone_gpkg_name: Name of training zone geopackage
        training_zone_layer: Layer name in training zone geopackage
        use_scheduler: Whether to use learning rate scheduler
        loss_type: Loss function choice:
            'ce' - CrossEntropyLoss (pixel-wise)
            'bce' - BCEWithLogitsLoss (pixel-wise)
            'dice' - Dice Loss (IoU-based, region overlap)
            'bce_dice' - Combined BCE + Dice (50/50 mix, RECOMMENDED for imbalanced data)
            'ce_dice' - Combined CE + Dice (50/50 mix)
        run_inference: Whether to run inference on test zones after training completes
    
    Returns:
        Dictionary with training state and paths
    """
    
    # Set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    # Create bands identifier for dataset name
    if bands_type == 'custom':
        if custom_bands_name:
            bands_id = custom_bands_name
        else:
            bands_id = 'CUSTOM'
    else:
        bands_id = bands_type.upper()
    
    # Dataset name: only bands type and reserves (reusable across hyperparameters)
    dataset_name = f"{bands_id}_{'_'.join(reserves)}"
    
    # Model name: includes all hyperparameters for unique identification
    loss_suffix = loss_type.upper().replace('_', '')
    model_name = f"MW{int(maire_weight)}_LR{str(lr).replace('.', '_')}_{loss_suffix}"
    
    # Define directory structure
    # Shared dataset directory (reused across different hyperparameters)
    dataset_dir = os.path.join(out_dir, 'datasets', dataset_name)
    training_dataset_dir = os.path.join(dataset_dir, 'training_dataset')
    
    # Model-specific output directory
    model_output_dir = os.path.join(out_dir, 'models', dataset_name, model_name)

    if os.path.exists(os.path.join(model_output_dir, 'best_model_maire_f1.pth')):
        print(f"Model already trained: {model_name} at {model_output_dir}")
        return {
            'state': None,
            'dataset_name': dataset_name,
            'model_name': model_name,
            'dataset_dir': dataset_dir,
            'training_dataset_dir': training_dataset_dir,
            'model_output_dir': model_output_dir,
            'in_channels': None,
            'loss_type': loss_type
        }
    else:
        shutil.rmtree(model_output_dir, ignore_errors=True)
    
    # Create training dataset if it doesn't exist
    if os.path.exists(training_dataset_dir):
        shutil.rmtree(training_dataset_dir)
    print(f"Creating training dataset for {dataset_name}...")
    
    # Temporary merge directory
    temp_merge_dir = os.path.join(dataset_dir, 'temp_merge')
    temp_images_dir = os.path.join(temp_merge_dir, f'aerial_{tile_size}')
    temp_labels_dir = os.path.join(temp_merge_dir, f'label_{tile_size}')
    makedirs(temp_images_dir)
    makedirs(temp_labels_dir)
    
    # Process each reserve
    for r in use(reserves):
        print(f"  Processing reserve: {r.name}")
        
        # Get label and training zone paths
        label_gpkg = os.path.join(r.GIS, label_gpkg_name)
        label_gdf = gpd.read_file(label_gpkg, layer=label_layer)
        training_zone_gpkg = os.path.join(r.GIS, training_zone_gpkg_name)
        
        # Select bands based on type
        bands = get_bands_for_reserve(r, bands_type, custom_bands_func)
        if bands_type == 'rgb':
            label_layer = 'maire_poly_rgb'
        
        # Rasterize labels
        rasterized_label = os.path.join(r.DS_UNET, f'maire_{label_layer}.tif')
        if not os.path.exists(rasterized_label):
            rasterize_vector(label_gdf, bands[0], rasterized_label, 8192, 
                        training_zone_gpkg, training_zone_layer)
        
        # Reserve-specific output directories (named by dataset, not full config)
        reserve_image_dir = os.path.join(r.DS_UNET, dataset_name, f'aerial_{tile_size}')
        reserve_label_dir = os.path.join(r.DS_UNET, dataset_name, f'label_{tile_size}')

        # Create tiles only if they don't exist
        if not os.path.exists(reserve_image_dir) or not os.path.exists(reserve_label_dir):

            makedirs(reserve_image_dir, exist_ok=True)
            makedirs(reserve_label_dir, exist_ok=True)
            
            # Tile images
            tile_image(
                gdf_path=training_zone_gpkg,
                layer=training_zone_layer,
                tif_paths=bands,
                output_dir=reserve_image_dir,
                file_prefix=f'{r.name if hasattr(r, "name") else "reserve"}_',
                tile_size=tile_size,
                clear_dir=True
            )
            
            # Clip labels
            clip_geotiffs(
                rasterized_label,
                reference_dir=reserve_image_dir,
                output_dir=reserve_label_dir
            )
        
        # Always link to temp merge directory (whether newly created or existing)
        for f in os.listdir(reserve_image_dir):
            src = os.path.join(reserve_image_dir, f)
            dst = os.path.join(temp_images_dir, f)
            if not os.path.exists(dst):
                os.link(src, dst)
        
        for f in os.listdir(reserve_label_dir):
            src = os.path.join(reserve_label_dir, f)
            dst = os.path.join(temp_labels_dir, f)
            if not os.path.exists(dst):
                os.link(src, dst)
    
    # === END OF RESERVE LOOP ===
    # All code below runs AFTER all reserves have been processed
    
    # Distribute combined dataset from all reserves
    print(f"Distributing combined dataset from {len(reserves)} reserves into train/valid splits...")
    distribute_files_with_target_balance(
        dataset_dir=temp_merge_dir,
        folders=['train', 'valid'],
        distribution=[0.8, 0.2],
        image_folder=f'aerial_{tile_size}',
        label_folder=f'label_{tile_size}',
        ignore_nomatch=False,
        out_dir=training_dataset_dir
    )
    
    # Clean up temp merge directory
    shutil.rmtree(temp_merge_dir)
    print("Training dataset created successfully!")
    
    # Calculate and save class distribution
    calculate_class_distribution(dataset_dir, training_dataset_dir, dataset_name)
    
    # Prepare for training
    print(f"Training model: {model_name}")
    makedirs(model_output_dir, delete_if_exists=True, exist_ok=False)
    
    # Get number of input channels
    in_channels = get_num_bands(list_files(training_dataset_dir, '.tif', True)[0])
    
    # Initialize model with appropriate output mode
    output_mode = 'binary' if 'bce' in loss_type or 'dice' in loss_type else 'multiclass'
    model = UNet(in_channels=in_channels, num_classes=len(ALL_CLASSES), output_mode=output_mode).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # Setup loss function
    if loss_type == 'bce':
        # BCEWithLogitsLoss includes sigmoid, more numerically stable
        # pos_weight applies extra weight to positive class (maire)
        pos_weight = torch.tensor([maire_weight]).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using BCEWithLogitsLoss with pos_weight={maire_weight}")
    
    elif loss_type == 'ce':
        criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
        print(f"Using CrossEntropyLoss with class_weights={CLASS_WEIGHTS.cpu().tolist()}")
    
    elif loss_type == 'dice':
        # Pure Dice loss - directly optimizes IoU-like metric
        # Apply weight to positive (maire) class for binary mode
        weight = torch.tensor([maire_weight]).to(DEVICE) if output_mode == 'binary' else CLASS_WEIGHTS
        criterion = DiceLoss(weight=weight)
        print(f"Using Weighted Dice Loss (IoU-based, weight={maire_weight if output_mode == 'binary' else CLASS_WEIGHTS.cpu().tolist()})")
    
    elif loss_type == 'dice_unweighted':
        # Unweighted Dice loss (treats all classes equally)
        criterion = DiceLoss()
        print(f"Using Unweighted Dice Loss (IoU-based, all classes equal)")
    
    elif loss_type == 'bce_dice':
        # Combined BCE + Dice (RECOMMENDED for imbalanced segmentation)
        pos_weight = torch.tensor([maire_weight]).to(DEVICE)
        bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5, bce_criterion=bce_criterion)
        print(f"Using Combined BCE+Dice Loss (50/50, pos_weight={maire_weight})")
    
    elif loss_type == 'ce_dice':
        # Combined CE + Dice
        ce_criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
        criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5, ce_criterion=ce_criterion)
        print(f"Using Combined CE+Dice Loss (50/50, class_weights={CLASS_WEIGHTS.cpu().tolist()})")
    
    else:
        raise ValueError(f"Invalid loss_type: {loss_type}. Choose from 'ce', 'bce', 'dice', 'dice_unweighted', 'bce_dice', 'ce_dice'")
    
    # Get datasets - pass appropriate loss_type to format masks
    # Dice and BCE-based losses need BCE-style masks
    dataset_loss_type = 'bce' if 'bce' in loss_type or 'dice' in loss_type else 'ce'
    train_images, train_masks, valid_images, valid_masks = get_images(root_path=training_dataset_dir)
    train_dataset, valid_dataset = get_dataset(
        train_images, train_masks, valid_images, valid_masks,
        ALL_CLASSES, ALL_CLASSES, LABEL_COLORS_LIST, img_size=None, loss_type=dataset_loss_type
    )
    train_dataloader, valid_dataloader = get_data_loaders(
        train_dataset, valid_dataset, batch_size=batch_size
    )
    
    # Setup scheduler with appropriate mode based on metric
    scheduler = None
    if use_scheduler:
        # mode='max' for IoU (higher is better), mode='min' for loss (lower is better)
        scheduler_mode = 'max' if scheduler_metric == 'iou' else 'min'
        scheduler = ReduceLROnPlateau(
            optimizer, mode=scheduler_mode, factor=0.5, patience=15, 
            threshold=0.00001, threshold_mode='rel', cooldown=0, 
            min_lr=0, eps=1e-08
        )
        print(f"Scheduler configured: metric={scheduler_metric}, mode={scheduler_mode}, patience=15")
    
    # Map loss_type to display name for dashboard
    loss_display_names = {
        'bce': 'WBCE',
        'ce': 'CE',
        'dice': 'Weighted Dice',
        'dice_unweighted': 'Dice',
        'bce_dice': 'WBCE+Dice',
        'ce_dice': 'CE+Dice'
    }
    loss_name = loss_display_names.get(loss_type, loss_type.upper())
    
    # Prepare test configuration for automated testing after training
    # Build test zones config for all reserves
    test_zones_config = []
    test_bands = None
    
    for r in use(reserves):
        # Test zone configuration
        test_zone_gpkg = os.path.join(r.GIS, training_zone_gpkg_name)
        
        # Build test_bands from first reserve (only once)
        if test_bands is None:
            test_bands = get_bands_for_reserve(r, bands_type, custom_bands_func)
        
        # Add unet_test_zone layer
        if os.path.exists(test_zone_gpkg):
            test_zones_config.append({
                'reserve': r.name if hasattr(r, 'name') else 'unknown',
                'gpkg_path': test_zone_gpkg,
                'layer': 'unet_test_zone',
                'zone_name': f'{r.name if hasattr(r, "name") else "unknown"}_test',
                'label_path': os.path.join(r.DS_UNET, f'maire_{label_layer}.tif')
            })
        
        # Add bbox layer (full extent)
        if os.path.exists(test_zone_gpkg):
            test_zones_config.append({
                'reserve': r.name if hasattr(r, 'name') else 'unknown',
                'gpkg_path': test_zone_gpkg,
                'layer': 'bbox',
                'zone_name': f'{r.name if hasattr(r, "name") else "unknown"}_bbox',
                'label_path': os.path.join(r.DS_UNET, f'maire_{label_layer}.tif')
            })
    
    test_config = {
        'test_zones': test_zones_config,
        'bands': test_bands,
        'output_mode': output_mode,
        'tile_size': tile_size
    }
    
    # Train model
    print(f"Starting training for {epochs} epochs...")
    state = train_model(
        model, train_dataset, train_dataloader, 
        valid_dataset, valid_dataloader, 
        optimizer, criterion, scheduler, epochs, model_output_dir,
        scheduler_metric=scheduler_metric,
        loss_name=loss_name,
        test_config=test_config
    )
    
    # Run inference on test zones after training completes
    if run_inference:
        run_inference_on_test_zones(
            reserves=reserves,
            bands_type=bands_type,
            dataset_name=dataset_name,
            model_name=model_name,
            model_base_dir=model_output_dir,
            test_layers=['unet_test_zone', 'bbox'],
            tile_size=tile_size,
            custom_bands_func=custom_bands_func
        )
        
        # Evaluate stitched predictions against vector ground truth
        print("\nEvaluating stitched predictions...")
        
        for r in use(reserves):
            label_gpkg = os.path.join(r.GIS, label_gpkg_name)
            zone_gpkg = os.path.join(r.GIS, training_zone_gpkg_name)
            ref_bands = get_bands_for_reserve(r, bands_type, custom_bands_func)
            reference_raster = ref_bands[0] if ref_bands else None
            
            for model_key in ['best_maire_f1', 'best_iou']:
                for zone_layer in ['unet_test_zone', 'bbox']:
                    # Look for reserve-specific prediction file
                    # Format: best_maire_f1_ESK_bbox.tif
                    pred_tif = os.path.join(model_output_dir, 'test_predictions', 
                                        f'{model_key}_{r.name}_{zone_layer}.tif')
                    
                    if os.path.exists(pred_tif) and reference_raster:
                        # Output CSV also reserve-specific
                        output_csv = os.path.join(model_output_dir, 'test_predictions',
                                                f'{model_key}_{r.name}_{zone_layer}_metrics.csv')
                        try:
                            evaluate_stitched_predictions(
                                prediction_tif=pred_tif,
                                label_gpkg=label_gpkg,
                                label_layer=label_layer,
                                zone_gpkg=zone_gpkg,
                                zone_layer=zone_layer,
                                reference_raster=reference_raster,
                                model_name=model_key,
                                zone_name=f'{r.name}_{zone_layer}',
                                output_csv=output_csv
                            )
                        except Exception as e:
                            print(f"  Warning: {r.name}/{zone_layer}/{model_key}: {e}")
    
    return {
        'state': state,
        'dataset_name': dataset_name,
        'model_name': model_name,
        'dataset_dir': dataset_dir,
        'training_dataset_dir': training_dataset_dir,
        'model_output_dir': model_output_dir,
        'in_channels': in_channels,
        'loss_type': loss_type
    }


import os
import shutil
import geopandas as gpd
from config.paths import use
from unet_2.preprocessing import tile_image
from unet_2.src.engine import make_predictions
from unet_2.postprocessing import copy_metadata, merge_geotiffs
from utils.helper_functions import list_files


def run_inference_on_test_zones(
    reserves,
    bands_type,
    dataset_name,
    model_name,
    model_base_dir,
    test_layers=['unet_test_zone', 'bbox'],
    tile_size=576,
    custom_bands_func=None
):
    """
    Run inference on test zones using trained models.
    
    Args:
        reserves: List of reserve codes
        bands_type: Type of bands ('ms_rel', 'rgb', etc.)
        dataset_name: Dataset identifier
        model_name: Model identifier
        model_base_dir: Base directory containing model checkpoints
        test_layers: List of layers to process (default: ['unet_test_zone', 'bbox'])
        tile_size: Size of image tiles
        custom_bands_func: Function to get custom bands (if bands_type='custom')
    
    Returns:
        dict: Mapping of (reserve, layer) -> prediction path
    """
    # Model paths with fallback hierarchy
    model_types_to_run = ['best_model_maire_f1', 'best_model_iou']
    
    print(f"\nRunning inference on test zones ({dataset_name}/{model_name})...")
    
    # Central output directory for all predictions
    central_preds_dir = os.path.join(model_base_dir, 'test_predictions')
    os.makedirs(central_preds_dir, exist_ok=True)
    
    prediction_paths = {}
    
    for model_type in model_types_to_run:
        model_path = os.path.join(model_base_dir, f'{model_type}.pth')
        
        if not os.path.exists(model_path):
            print(f"  Skipping {model_type} - not found")
            continue
        
        # Short name for output files (remove 'best_model_' prefix)
        model_short = model_type.replace('best_model_', 'best_')
        
        for r in use(reserves):
            training_zone_gpkg = os.path.join(r.GIS, 'bbox.gpkg')
            bands = get_bands_for_reserve(r, bands_type, custom_bands_func)
            
            print(f"  {r.name} ({model_short}): ", end="")
            
            # Process each test layer
            for layer in test_layers:
                # Working directories for this reserve/layer
                out_dir = os.path.join(r.DS_UNET, dataset_name, model_name, layer)
                tiles_dir = os.path.join(out_dir, f'aerial_{tile_size}')
                preds_dir = os.path.join(out_dir, 'predictions')
                pred_tiles_dir = os.path.join(preds_dir, 'tiles')
                
                os.makedirs(tiles_dir, exist_ok=True)
                os.makedirs(preds_dir, exist_ok=True)

                # Tile images for this layer
                try:
                    tile_image(
                        gdf_path=training_zone_gpkg,
                        layer=layer,
                        tif_paths=bands,
                        output_dir=tiles_dir,
                        file_prefix=f'{r.name}_{layer}_',
                        tile_size=tile_size,
                        clear_dir=True
                    )
                except Exception as e:
                    print(f"{layer}(tile error) ", end="")
                    continue

                # Run predictions
                try:
                    make_predictions(tiles_dir, model_path, pred_tiles_dir, output_mode='binary')
                except Exception as e:
                    print(f"{layer}(inference error) ", end="")
                    continue
                
                # Copy metadata and merge tiles
                # Save to reserve-specific location first
                reserve_out_tif = os.path.join(preds_dir, f'{model_short}_{layer}.tif')
                try:
                    copy_metadata(rgb_dir=tiles_dir, gray_dir=pred_tiles_dir)
                    merge_geotiffs(pred_tiles_dir, reserve_out_tif)
                except Exception as e:
                    print(f"{layer}(stitch error) ", end="")
                    continue
                
                # Copy to central location with reserve-specific name
                # Format: best_maire_f1_ESK_bbox.tif
                central_out_tif = os.path.join(central_preds_dir, f'{model_short}_{r.name}_{layer}.tif')
                shutil.copy2(reserve_out_tif, central_out_tif)
                prediction_paths[(r.name, model_short, layer)] = central_out_tif
                
                print(f"{layer}✓ ", end="")
            
            print()  # New line after reserve

    print("Inference complete.")
    return prediction_paths