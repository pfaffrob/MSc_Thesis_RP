"""
U-Net training and evaluation engine.
Implements training loops, loss functions, metrics calculation, and model evaluation.
"""

import torch

from tqdm import tqdm
from .utils import draw_translucent_seg_maps, save_model, SaveBestModel, SaveBestModelIOU, SaveBestModelMaireF1, update_plots, update_csv, get_segment_labels, draw_segmentation_map
from .metrics import IOUEval
import torch
import torch.nn as nn
import torch.optim as optim
import os
import cv2
import csv
import matplotlib.pyplot as plt
from .model import UNet
import numpy as np
import tifffile as tiff

from .config import ALL_CLASSES, LABEL_COLORS_LIST, DEVICE


classes_to_train = ALL_CLASSES


def calculate_metrics(outputs, targets, threshold=0.5):
    """
    Calculate Precision, Recall, F1 for binary segmentation.
    For binary maire detection, calculates both overall and maire-specific metrics.
    
    Args:
        outputs: Model outputs (logits for BCE mode)
        targets: Ground truth labels
        threshold: Threshold for binary classification
    
    Returns:
        Dictionary with overall and maire-specific metrics:
        - precision, recall, f1 (overall)
        - maire_precision, maire_recall, maire_f1, maire_iou (maire class only)
        - TP, FP, FN, TN
    """
    # Get predictions
    if outputs.shape[1] == 1:  # Binary mode
        preds = (torch.sigmoid(outputs) > threshold).float()
        preds = preds.view(-1)
        targets = targets.view(-1)
    else:  # Multiclass mode - calculate for each class
        preds = torch.argmax(outputs, dim=1)
        preds = preds.view(-1)
        targets = targets.view(-1)
    
    # Confusion matrix components (maire = positive class = 1)
    TP = ((preds == 1) & (targets == 1)).sum().float()  # Maire correctly predicted
    FP = ((preds == 1) & (targets == 0)).sum().float()  # Background predicted as maire
    FN = ((preds == 0) & (targets == 1)).sum().float()  # Maire predicted as background
    TN = ((preds == 0) & (targets == 0)).sum().float()  # Background correctly predicted
    
    # Calculate metrics with epsilon to avoid division by zero
    eps = 1e-8
    
    # Overall metrics (macro-averaged)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    
    # Maire-specific metrics (positive class metrics)
    maire_precision = TP / (TP + FP + eps)  # Of predicted maire pixels, % correct
    maire_recall = TP / (TP + FN + eps)     # Of actual maire pixels, % detected
    maire_f1 = 2 * (maire_precision * maire_recall) / (maire_precision + maire_recall + eps)
    maire_iou = TP / (TP + FP + FN + eps)   # Intersection over Union for maire
    
    # Background-specific metrics (negative class metrics)
    background_precision = TN / (TN + FN + eps)
    background_recall = TN / (TN + FP + eps)
    
    return {
        # Overall metrics
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        # Maire-specific metrics (most important for imbalanced data)
        'maire_precision': maire_precision.item(),
        'maire_recall': maire_recall.item(),
        'maire_f1': maire_f1.item(),
        'maire_iou': maire_iou.item(),
        # Background-specific metrics
        'background_precision': background_precision.item(),
        'background_recall': background_recall.item(),
        # Confusion matrix
        'TP': TP.item(),
        'FP': FP.item(),
        'FN': FN.item(),
        'TN': TN.item()
    }


class DiceLoss(nn.Module):
    """
    Dice Loss for binary or multiclass segmentation.
    Dice coefficient = 2*|X∩Y| / (|X|+|Y|)
    Dice Loss = 1 - Dice coefficient
    
    Very similar to IoU but differentiable and more stable for gradients.
    """
    def __init__(self, smooth=1.0, weight=None):
        """
        Args:
            smooth: Smoothing constant to avoid division by zero
            weight: Class weights (tensor). For binary: single value for positive class.
                   For multiclass: tensor of shape (num_classes,)
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.weight = weight
    
    def forward(self, outputs, targets):
        # For BCE mode: outputs shape (B, 1, H, W), targets shape (B, 1, H, W)
        # For CE mode: outputs shape (B, C, H, W), targets shape (B, H, W)
        
        if outputs.shape[1] == 1:  # Binary segmentation (BCE mode)
            # Apply sigmoid to get probabilities
            outputs = torch.sigmoid(outputs)
            # Flatten
            outputs = outputs.view(-1)
            targets = targets.view(-1)
            
            intersection = (outputs * targets).sum()
            dice = (2. * intersection + self.smooth) / (outputs.sum() + targets.sum() + self.smooth)
            
            # Apply weight to positive class if provided
            if self.weight is not None:
                # Weight the dice loss for the positive class
                # Higher weight means more penalty when dice is low
                dice_loss = 1 - dice
                if isinstance(self.weight, torch.Tensor):
                    weight_val = self.weight.item() if self.weight.numel() == 1 else self.weight[0].item()
                else:
                    weight_val = self.weight
                return dice_loss * weight_val
            
            return 1 - dice
            
        else:  # Multiclass segmentation (CE mode)
            # Apply softmax to get probabilities
            outputs = torch.softmax(outputs, dim=1)
            
            # Convert targets to one-hot encoding
            num_classes = outputs.shape[1]
            targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes)
            targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
            
            # Flatten spatial dimensions
            outputs = outputs.view(outputs.shape[0], outputs.shape[1], -1)
            targets_one_hot = targets_one_hot.view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)
            
            # Calculate Dice per class
            intersection = (outputs * targets_one_hot).sum(dim=2)
            dice_per_class = (2. * intersection + self.smooth) / (outputs.sum(dim=2) + targets_one_hot.sum(dim=2) + self.smooth)
            
            # Apply class weights if provided
            if self.weight is not None:
                # Weight each class's dice coefficient
                dice_loss_per_class = 1 - dice_per_class
                weighted_loss = (dice_loss_per_class * self.weight.unsqueeze(0)).mean()
                return weighted_loss
            
            dice = dice_per_class.mean()
            return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combination of BCE/CE loss with Dice loss.
    Combines pixel-wise accuracy (BCE/CE) with region overlap (Dice).
    
    For binary segmentation (outputs.shape[1] == 1):
        l_WBCE+Dice = α·l_WBCE + (1-α)·l_Dice
    
    For multiclass segmentation:
        l_CE+Dice = α·l_CE + (1-α)·l_Dice
    """
    def __init__(self, ce_weight=0.5, dice_weight=0.5, bce_criterion=None, ce_criterion=None):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight  # α (alpha)
        self.dice_weight = dice_weight  # (1-α)
        self.bce_criterion = bce_criterion
        self.ce_criterion = ce_criterion
        # Use unweighted Dice loss here since weighting is handled by the base criterion
        self.dice_loss = DiceLoss(weight=None)
    
    def forward(self, outputs, targets):
        # Determine if binary or multiclass based on output shape
        is_binary = (outputs.shape[1] == 1)
        
        # Calculate base loss (WBCE or CE)
        if self.bce_criterion is not None:
            base_loss = self.bce_criterion(outputs, targets)
        elif self.ce_criterion is not None:
            base_loss = self.ce_criterion(outputs, targets)
        else:
            raise ValueError("Either bce_criterion or ce_criterion must be provided")
        
        # Calculate Dice loss (unweighted)
        dice_loss = self.dice_loss(outputs, targets)
        
        # Combine losses: α·l_base + (1-α)·l_Dice
        combined = self.ce_weight * base_loss + self.dice_weight * dice_loss
        
        return combined

def train(
    model,
    train_dataset,
    train_dataloader,
    optimizer,
    criterion,
    classes_to_train
):
    print('Training')
    model.train()
    train_running_loss = 0.0
    # Calculate the number of batches.
    num_batches = int(len(train_dataset)/train_dataloader.batch_size)
    prog_bar = tqdm(train_dataloader, total=num_batches, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    counter = 0 # to keep track of batch counter
    num_classes = len(classes_to_train)
    iou_eval = IOUEval(num_classes)
    
    # Accumulate metrics across batches
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_maire_precision = 0.0
    total_maire_recall = 0.0
    total_maire_f1 = 0.0
    total_maire_iou = 0.0

    for i, data in enumerate(prog_bar):
        counter += 1
        data, target = data[0].to(DEVICE), data[1].to(DEVICE)
        optimizer.zero_grad()
        outputs = model(data)

        ##### BATCH-WISE LOSS #####
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        ###########################

        ##### BACKPROPAGATION AND PARAMETER UPDATION #####
        loss.backward()
        optimizer.step()
        ##################################################

        # Handle both BCE (single channel) and CE (multi-channel) outputs for metrics
        if outputs.shape[1] == 1:  # BCE mode: single channel
            # Apply sigmoid and threshold to get binary predictions
            preds = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)
            targets = target.long().squeeze(1) if target.dim() == 4 else target.long()
        else:  # CE mode: multi-channel
            preds = outputs.max(1)[1].data
            targets = target.data
        
        iou_eval.addBatch(preds, targets)
        
        # Calculate precision, recall, F1 for this batch
        metrics = calculate_metrics(outputs, target)
        total_precision += metrics['precision']
        total_recall += metrics['recall']
        total_f1 += metrics['f1']
        total_maire_precision += metrics['maire_precision']
        total_maire_recall += metrics['maire_recall']
        total_maire_f1 += metrics['maire_f1']
        total_maire_iou += metrics['maire_iou']
        
    ##### PER EPOCH LOSS #####
    train_loss = train_running_loss / counter
    ##########################
    overall_acc, per_class_acc, per_class_iu, mIOU = iou_eval.getMetric()
    
    # Average metrics across batches
    avg_precision = total_precision / counter
    avg_recall = total_recall / counter
    avg_f1 = total_f1 / counter
    avg_maire_precision = total_maire_precision / counter
    avg_maire_recall = total_maire_recall / counter
    avg_maire_f1 = total_maire_f1 / counter
    avg_maire_iou = total_maire_iou / counter

    return train_loss, overall_acc, mIOU, per_class_iu, per_class_acc, avg_precision, avg_recall, avg_f1, avg_maire_precision, avg_maire_recall, avg_maire_f1, avg_maire_iou

def validate(
    model,
    valid_dataset,
    valid_dataloader,
    criterion,
    classes_to_train,
    label_colors_list,
    epoch,
    all_classes,
    save_dir,
    reference_sample_data=None,
    reference_sample_target=None
):
    print('Validating')
    model.eval()
    valid_running_loss = 0.0
    # Calculate the number of batches.
    num_batches = int(len(valid_dataset)/valid_dataloader.batch_size)
    num_classes = len(classes_to_train)
    iou_eval = IOUEval(num_classes)
    
    # Accumulate metrics across batches
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_maire_precision = 0.0
    total_maire_recall = 0.0
    total_maire_f1 = 0.0
    total_maire_iou = 0.0

    with torch.no_grad():
        prog_bar = tqdm(valid_dataloader, total=num_batches, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        counter = 0 # To keep track of batch counter.
        for i, data in enumerate(prog_bar):
            counter += 1
            data, target = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = model(data)

            ##### BATCH-WISE LOSS #####
            loss = criterion(outputs, target)
            valid_running_loss += loss.item()
            ###########################

            # Handle both BCE (single channel) and CE (multi-channel) outputs for metrics
            if outputs.shape[1] == 1:  # BCE mode: single channel
                # Apply sigmoid and threshold to get binary predictions
                preds = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)
                targets = target.long().squeeze(1) if target.dim() == 4 else target.long()
            else:  # CE mode: multi-channel
                preds = outputs.max(1)[1].data
                targets = target.data
            
            iou_eval.addBatch(preds, targets)
            
            # Calculate precision, recall, F1 for this batch
            metrics = calculate_metrics(outputs, target)
            total_precision += metrics['precision']
            total_recall += metrics['recall']
            total_f1 += metrics['f1']
            total_maire_precision += metrics['maire_precision']
            total_maire_recall += metrics['maire_recall']
            total_maire_f1 += metrics['maire_f1']
            total_maire_iou += metrics['maire_iou']
        
        # After all batches, use the reference sample for visualization if provided
        if reference_sample_data is not None and reference_sample_target is not None:
            # Get prediction for the reference sample
            reference_output = model(reference_sample_data)
            
            draw_translucent_seg_maps(
                reference_sample_data, 
                reference_output, 
                epoch, 
                0,  # batch index 0 since it's a single sample
                save_dir, 
                label_colors_list,
            )
            
            # Save ground truth mask once in the first epoch
            if epoch == 0:
                import cv2
                # Convert target to visualization format
                if reference_sample_target.dim() == 4:  # BCE mode: (1, 1, H, W)
                    mask_np = reference_sample_target.squeeze().cpu().numpy()
                else:  # CE mode: (1, H, W)
                    mask_np = reference_sample_target.squeeze().cpu().numpy()
                
                # Scale to 0-255 for visualization
                mask_vis = (mask_np * 255).astype('uint8')
                mask_path = os.path.join(save_dir, 'ground_truth_mask.png')
                cv2.imwrite(mask_path, mask_vis)
                print(f"Saved ground truth mask to {mask_path}")
    
    ##### PER EPOCH LOSS #####
    valid_loss = valid_running_loss / counter
    ##########################
    overall_acc, per_class_acc, per_class_iu, mIOU = iou_eval.getMetric()
    
    # Average metrics across batches
    avg_precision = total_precision / counter
    avg_recall = total_recall / counter
    avg_f1 = total_f1 / counter
    avg_maire_precision = total_maire_precision / counter
    avg_maire_recall = total_maire_recall / counter
    avg_maire_f1 = total_maire_f1 / counter
    avg_maire_iou = total_maire_iou / counter
    
    return valid_loss, overall_acc, mIOU, per_class_iu, per_class_acc, avg_precision, avg_recall, avg_f1, avg_maire_precision, avg_maire_recall, avg_maire_f1, avg_maire_iou

save_best_model = SaveBestModel()
save_best_iou = SaveBestModelIOU()
save_best_maire_f1 = SaveBestModelMaireF1()

def train_model(model, train_dataset, train_dataloader, valid_dataset, valid_dataloader, optimizer, criterion, scheduler, epochs, out_dir, use_scheduler=True, patience=15, state=None, scheduler_metric='loss', loss_name='Loss', test_config=None):
    """
    Train model and optionally run test set evaluation after training.
    
    Args:
        scheduler_metric: 'loss' to reduce LR on validation loss plateau (default),
                         'iou' to reduce LR on validation IoU plateau (better for imbalanced segmentation)
        test_config: Optional dict with keys:
            - 'test_zones': List of dicts with 'reserve', 'gpkg_path', 'layer', 'zone_name', 'label_path'
            - 'bands': List of band paths for test tiling
            - 'output_mode': 'binary' or 'multiclass'
            - 'tile_size': Size of tiles for inference
    """
    # Ensure the output directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Initialize CSV file
    csv_file = os.path.join(out_dir, 'training_metrics.csv')
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Create header with per-class mIOU columns
            header = ['Epoch', 'Train Loss', 'Valid Loss', 'Train Accuracy', 'Valid Accuracy', 
                     'Train Precision', 'Valid Precision', 'Train Recall', 'Valid Recall', 
                     'Train F1', 'Valid F1', 
                     'Train Maire Precision', 'Valid Maire Precision', 
                     'Train Maire Recall', 'Valid Maire Recall', 
                     'Train Maire F1', 'Valid Maire F1', 
                     'Train Maire IoU', 'Valid Maire IoU', 
                     'Train mIOU', 'Valid mIOU', 'Learning Rate']
            # Add per-class mIOU columns for train and valid
            for class_name in classes_to_train:
                header.append(f'Train mIOU {class_name}')
            for class_name in classes_to_train:
                header.append(f'Valid mIOU {class_name}')
            # Add per-class accuracy columns for train and valid
            for class_name in classes_to_train:
                header.append(f'Train Acc {class_name}')
            for class_name in classes_to_train:
                header.append(f'Valid Acc {class_name}')
            writer.writerow(header)

    # Load persisted best IoU if available so resumed runs don't lose the
    # best-IoU history. This file is written by SaveBestModelIOU upon saving.
    best_iou_file = os.path.join(out_dir, 'best_iou.txt')
    if os.path.exists(best_iou_file):
        try:
            with open(best_iou_file, 'r') as f:
                val = float(f.read().strip())
                save_best_iou.best_iou = val
                print(f"Loaded persisted best IoU: {val}")
        except Exception as e:
            print(f"Could not load persisted best IoU: {e}")


    # Initialize training state
    if state:
        start_epoch, train_loss, valid_loss, train_pix_acc, valid_pix_acc, train_miou, valid_miou, lr_history = state
        # Initialize new metrics lists if not in state
        train_precision, valid_precision = [], []
        train_recall, valid_recall = [], []
        train_f1, valid_f1 = [], []
        train_maire_precision, valid_maire_precision = [], []
        train_maire_recall, valid_maire_recall = [], []
        train_maire_f1, valid_maire_f1 = [], []
        train_maire_iou, valid_maire_iou = [], []
    else:
        start_epoch = 0
        train_loss, train_pix_acc, train_miou = [], [], []
        valid_loss, valid_pix_acc, valid_miou = [], [], []
        train_precision, valid_precision = [], []
        train_recall, valid_recall = [], []
        train_f1, valid_f1 = [], []
        train_maire_precision, valid_maire_precision = [], []
        train_maire_recall, valid_maire_recall = [], []
        train_maire_f1, valid_maire_f1 = [], []
        train_maire_iou, valid_maire_iou = [], []
        lr_history = []

    # Early stopping parameters - only start monitoring from epoch 5
    best_train_loss = float('inf')
    epochs_no_improve = 0
    baseline_epoch = 5

    out_dir_valid_preds = os.path.join(out_dir, 'valid_preds')
    os.makedirs(out_dir_valid_preds, exist_ok=True)

    # Select the most balanced validation sample once (before training)
    print("Selecting most balanced validation sample for visualization...")
    reference_sample_data = None
    reference_sample_target = None
    max_positive_ratio = 0.0
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(valid_dataloader):
            images = data[0].to(DEVICE)
            masks = data[1].to(DEVICE)
            
            # Check each sample in the batch for the most balanced class distribution
            for j in range(images.shape[0]):
                sample_mask = masks[j]
                # Calculate the ratio of positive class (maire) pixels
                if sample_mask.dim() == 3:  # CE mode: (C, H, W)
                    maire_pixels = (sample_mask[1] > 0).sum().item() if sample_mask.shape[0] > 1 else (sample_mask > 0).sum().item()
                elif sample_mask.dim() == 2:  # CE mode without batch: (H, W)
                    maire_pixels = (sample_mask == 1).sum().item()
                else:  # BCE mode: (1, H, W)
                    maire_pixels = (sample_mask > 0.5).sum().item()
                
                total_pixels = sample_mask.numel() if sample_mask.dim() <= 2 else sample_mask[0].numel()
                positive_ratio = maire_pixels / total_pixels if total_pixels > 0 else 0.0
                
                if positive_ratio > max_positive_ratio:
                    max_positive_ratio = positive_ratio
                    reference_sample_data = images[j:j+1].clone()  # Keep batch dimension
                    reference_sample_target = masks[j:j+1].clone()
    
    print(f"Selected reference sample with {max_positive_ratio*100:.2f}% maire pixels")
    
    # Enable interactive mode
    plt.ion()

    # Initialize the plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 14), dpi = 300)

    for epoch in range(start_epoch, epochs):
        print(f"EPOCH: {epoch + 1}")
        train_epoch_loss, train_epoch_pixacc, train_epoch_miou, train_per_class_iu, train_per_class_acc, train_epoch_precision, train_epoch_recall, train_epoch_f1, train_epoch_maire_precision, train_epoch_maire_recall, train_epoch_maire_f1, train_epoch_maire_iou = train(
            model, train_dataset, train_dataloader, optimizer, criterion, classes_to_train
        )
        valid_epoch_loss, valid_epoch_pixacc, valid_epoch_miou, valid_per_class_iu, valid_per_class_acc, valid_epoch_precision, valid_epoch_recall, valid_epoch_f1, valid_epoch_maire_precision, valid_epoch_maire_recall, valid_epoch_maire_f1, valid_epoch_maire_iou = validate(
            model, valid_dataset, valid_dataloader, criterion, classes_to_train, LABEL_COLORS_LIST,
            epoch, ALL_CLASSES, save_dir=out_dir_valid_preds, 
            reference_sample_data=reference_sample_data, reference_sample_target=reference_sample_target
        )
        train_loss.append(train_epoch_loss)
        train_pix_acc.append(train_epoch_pixacc)
        train_miou.append(train_epoch_miou)
        train_precision.append(train_epoch_precision)
        train_recall.append(train_epoch_recall)
        train_f1.append(train_epoch_f1)
        train_maire_precision.append(train_epoch_maire_precision)
        train_maire_recall.append(train_epoch_maire_recall)
        train_maire_f1.append(train_epoch_maire_f1)
        train_maire_iou.append(train_epoch_maire_iou)
        valid_loss.append(valid_epoch_loss)
        valid_pix_acc.append(valid_epoch_pixacc)
        valid_miou.append(valid_epoch_miou)
        valid_precision.append(valid_epoch_precision)
        valid_recall.append(valid_epoch_recall)
        valid_f1.append(valid_epoch_f1)
        valid_maire_precision.append(valid_epoch_maire_precision)
        valid_maire_recall.append(valid_epoch_maire_recall)
        valid_maire_f1.append(valid_epoch_maire_f1)
        valid_maire_iou.append(valid_epoch_maire_iou)

        save_best_model(valid_epoch_loss, epoch, model, out_dir, name='model_loss')
        save_best_iou(valid_epoch_miou, epoch, model, out_dir, name='model_iou')
        save_best_maire_f1(valid_epoch_maire_f1, epoch, model, out_dir, name='model_maire_f1')

        print(f"Train Epoch Loss: {train_epoch_loss:.4f}, Train Epoch PixAcc: {train_epoch_pixacc:.4f}, Train Epoch mIOU: {train_epoch_miou:.4f}")
        print(f"Train Overall - Precision: {train_epoch_precision:.4f}, Recall: {train_epoch_recall:.4f}, F1: {train_epoch_f1:.4f}")
        print(f"Train Maire   - Precision: {train_epoch_maire_precision:.4f}, Recall: {train_epoch_maire_recall:.4f}, F1: {train_epoch_maire_f1:.4f}, IoU: {train_epoch_maire_iou:.4f}")
        print(f"Valid Epoch Loss: {valid_epoch_loss:.4f}, Valid Epoch PixAcc: {valid_epoch_pixacc:.4f}, Valid Epoch mIOU: {valid_epoch_miou:.4f}")
        print(f"Valid Overall - Precision: {valid_epoch_precision:.4f}, Recall: {valid_epoch_recall:.4f}, F1: {valid_epoch_f1:.4f}")
        print(f"Valid Maire   - Precision: {valid_epoch_maire_precision:.4f}, Recall: {valid_epoch_maire_recall:.4f}, F1: {valid_epoch_maire_f1:.4f}, IoU: {valid_epoch_maire_iou:.4f}")
        
        if use_scheduler:
            # Step scheduler based on chosen metric (only after baseline epoch)
            if epoch + 1 >= baseline_epoch:
                if scheduler_metric == 'iou':
                    scheduler.step(valid_miou[-1])
                else:
                    scheduler.step(valid_loss[-1])
            current_lr = scheduler.get_last_lr()  # Extract the learning rate value
            lr_history.append(current_lr[-1])
            print(f"Current Learning Rate: {current_lr[-1]}")
        else:
            current_lr = optimizer.param_groups['lr']
            lr_history.append(current_lr[-1])
            print(f"Current Learning Rate: {current_lr[-1]}")

        # Update the plots
        update_plots(fig, ax1, ax2, ax3, ax4, train_f1, valid_f1, train_loss, valid_loss, train_miou, valid_miou, lr_history, epochs, loss_name)
        
        # Save the plot for the current epoch
        fig.savefig(os.path.join(out_dir, 'training_dashboard.png'), dpi=300)

        # Append metrics to CSV file
        # Update CSV file with the new epoch data
        update_csv(csv_file, epoch, train_epoch_loss, valid_epoch_loss, train_epoch_pixacc, valid_epoch_pixacc, 
                  train_epoch_precision, valid_epoch_precision, train_epoch_recall, valid_epoch_recall,
                  train_epoch_f1, valid_epoch_f1, 
                  train_epoch_maire_precision, valid_epoch_maire_precision,
                  train_epoch_maire_recall, valid_epoch_maire_recall,
                  train_epoch_maire_f1, valid_epoch_maire_f1,
                  train_epoch_maire_iou, valid_epoch_maire_iou,
                  train_epoch_miou, valid_epoch_miou, lr_history, 
                  train_per_class_iu, valid_per_class_iu, train_per_class_acc, valid_per_class_acc)


        # Save the model
        state = (epoch +1, train_loss, valid_loss, train_pix_acc, valid_pix_acc, train_miou, valid_miou, lr_history)
        #save_model(model, optimizer, criterion, out_dir, state)
        save_model(model, optimizer, criterion, out_dir, state, name='model')
        # Check for early stopping (only after baseline epoch)
        if epoch + 1 >= baseline_epoch:
            if train_epoch_loss < best_train_loss:
                best_train_loss = train_epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    print('TRAINING COMPLETE')
    
    # Run test set evaluation if configured
    if test_config is not None:
        print("\n" + "="*80)
        print("Starting test set evaluation...")
        print("="*80 + "\n")
        
        # Only evaluate best maire F1 and best IoU models (exclude best loss)
        model_paths = {
            'best_maire_f1': os.path.join(out_dir, 'best_model_maire_f1.pth'),
            'best_iou': os.path.join(out_dir, 'best_model_iou.pth')
        }
        
        test_results = evaluate_test_set(
            model_paths=model_paths,
            test_zones_config=test_config['test_zones'],
            bands=test_config['bands'],
            out_dir=out_dir,
            output_mode=test_config.get('output_mode', 'binary'),
            tile_size=test_config.get('tile_size', 576)
        )
        
        print("\nTest set evaluation complete!")
        print(f"Results saved to: {os.path.join(out_dir, 'test_metrics.csv')}")

    # Return the state
    return state



import rasterio
import torch
import os
import numpy as np
from tqdm import tqdm
from raster.utils import get_num_bands
from utils.helper_functions import list_files
import rasterio

def make_predictions(input_dir, model_path, out_dir, imgsz=None, output_mode='binary'):
    """
    Generate predictions for all images in input_dir using the specified model.
    
    Args:
        input_dir: Directory containing input images (.tif files)
        model_path: Path to the trained model checkpoint
        out_dir: Output directory for predictions
        imgsz: Optional resize dimension
        output_mode: 'binary' for BCE-based models, 'multiclass' for CE-based models
    """
    os.makedirs(out_dir, exist_ok=True)

    # Get number of bands from first image
    first_image_path = list_files(input_dir, '.tif', True)[0]
    in_channels = get_num_bands(first_image_path)

    # Load model with appropriate output mode
    model = UNet(in_channels=in_channels, num_classes=len(ALL_CLASSES), output_mode=output_mode)
    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(DEVICE)

    all_image_paths = [f for f in os.listdir(input_dir) if f.endswith('.tif')]
    for image_path in tqdm(all_image_paths, desc="Making predictions"):
        image_full_path = os.path.join(input_dir, image_path)

        # Read image using rasterio
        with rasterio.open(image_full_path) as src:
            image = src.read().astype('float32')
            image = image / (255.0 if src.dtypes[0] == 'uint8' else 65535.0)
            image = np.transpose(image, (1, 2, 0))  # (H, W, C)

        if imgsz is not None:
            image = cv2.resize(image, (int(imgsz), int(imgsz)))

        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).to(DEVICE)

        # Forward pass
        with torch.no_grad():
            outputs = model(image_tensor.unsqueeze(0))
            
            # Handle different output modes
            if output_mode == 'binary':
                # Apply sigmoid and threshold for binary segmentation
                mask = (torch.sigmoid(outputs.squeeze()) > 0.5).cpu().numpy().astype(np.uint8)
            else:
                # Argmax for multiclass segmentation
                mask = torch.argmax(outputs.squeeze(), dim=0).cpu().numpy().astype(np.uint8)

        # Save mask using rasterio
        output_path = os.path.join(out_dir, os.path.splitext(image_path)[0] + '.tif')
        save_mask_with_rasterio(mask, image_full_path, output_path)

def save_mask_with_rasterio(mask, reference_path, output_path):
    with rasterio.open(reference_path) as src:
        meta = src.meta.copy()
        meta.update({
            'count': 1,
            'dtype': 'uint8',
            'compress': 'lzw',
            'nodata': 255,
            'crs': 'EPSG:2193'
        })
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(mask, 1)


def evaluate_test_set(model_paths, test_zones_config, bands, out_dir, output_mode='binary', tile_size=576):
    """
    Run inference on test zones and calculate metrics for maire detection.
    Saves stitched prediction maps for each model and zone.
    
    Args:
        model_paths: Dictionary with keys 'maire_f1', 'iou' pointing to model checkpoint paths
        test_zones_config: List of dicts with 'reserve', 'gpkg_path', 'layer', 'zone_name'
        bands: List of band paths for creating test tiles
        out_dir: Output directory for test results
        output_mode: 'binary' or 'multiclass'
        tile_size: Size of tiles for inference
        
    Returns:
        Dictionary with test metrics for each model and zone
    """
    import geopandas as gpd
    import shutil
    from PIL import Image
    from unet_2.postprocessing import copy_metadata, merge_geotiffs
    
    print("\n" + "="*80)
    print("RUNNING TEST SET EVALUATION")
    print("="*80)
    print(f"Number of models to evaluate: {len(model_paths)}")
    print(f"Models: {list(model_paths.keys())}")
    print(f"Number of test zones: {len(test_zones_config)}")
    print(f"Test zones: {[z['zone_name'] for z in test_zones_config]}")
    print(f"\nDetailed test zone configuration:")
    for idx, zone in enumerate(test_zones_config, 1):
        print(f"  Zone {idx}: {zone['zone_name']}")
        print(f"    Reserve: {zone['reserve']}")
        print(f"    Layer: {zone['layer']}")
        print(f"    GPKG: {zone['gpkg_path']}")
    print(f"\nOutput mode: {output_mode}")
    print(f"Tile size: {tile_size}")
    print("="*80)
    
    test_results = {}
    csv_file = os.path.join(out_dir, 'test_metrics.csv')
    
    # Create output directory for stitched predictions
    predictions_out_dir = os.path.join(out_dir, 'test_predictions')
    os.makedirs(predictions_out_dir, exist_ok=True)
    print(f"Stitched predictions will be saved to: {predictions_out_dir}\n")
    
    # Create CSV header
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Test Zone', 'mIoU', 'Maire Precision', 'Maire Recall', 'Maire F1', 'Maire IoU'])
    print(f"Test metrics will be saved to: {csv_file}\n")
    
    for model_idx, (model_name, model_path) in enumerate(model_paths.items(), 1):
        print(f"\n{'='*80}")
        print(f"MODEL {model_idx}/{len(model_paths)}: {model_name}")
        print(f"Model path: {model_path}")
        print(f"{'='*80}")
        
        if not os.path.exists(model_path):
            print(f"❌ SKIPPING {model_name} - model file not found!")
            print(f"   Expected path: {model_path}")
            continue
        
        print(f"✓ Model file found: {os.path.basename(model_path)}")
        test_results[model_name] = {}
        
        print(f"\nStarting to process {len(test_zones_config)} test zones for model {model_name}...")
        
        for zone_idx, zone_config in enumerate(test_zones_config, 1):
            try:
                print(f"\n🔄 STARTING Zone {zone_idx}/{len(test_zones_config)}: {zone_config['zone_name']}")
                
                zone_name = zone_config['zone_name']
                reserve = zone_config['reserve']
                gpkg_path = zone_config['gpkg_path']
                layer = zone_config['layer']
                label_path = zone_config['label_path']
            
                print(f"\n{'-'*80}")
                print(f"TEST ZONE {zone_idx}/{len(test_zones_config)}: {zone_name}")
                print(f"  Reserve: {reserve}")
                print(f"  Layer: {layer}")
                print(f"  GPKG: {os.path.basename(gpkg_path)}")
                print(f"  Label: {os.path.basename(label_path)}")
                print(f"{'-'*80}")
                
                # Create temporary directories for this test zone
                temp_test_dir = os.path.join(out_dir, f'temp_test_{zone_name}_{model_name}')
                test_images_dir = os.path.join(temp_test_dir, f'aerial_{tile_size}')
                test_labels_dir = os.path.join(temp_test_dir, f'label_{tile_size}')
                test_preds_dir = os.path.join(temp_test_dir, 'predictions')
                
                # CRITICAL FIX: Force-delete temp directory to ensure clean state
                # This prevents reusing tiles from previous runs with different zones
                if os.path.exists(temp_test_dir):
                    print(f"  🧹 Removing old temporary directory: {temp_test_dir}")
                    shutil.rmtree(temp_test_dir)
                
                print(f"Creating fresh temporary directories:")
                print(f"  Images: {test_images_dir}")
                print(f"  Labels: {test_labels_dir}")
                print(f"  Predictions: {test_preds_dir}")
                
                os.makedirs(test_images_dir, exist_ok=False)
                os.makedirs(test_labels_dir, exist_ok=False)
            
                # Import tiling functions
                from unet_2.preprocessing import tile_image, clip_geotiffs
                
                # Tile images for test zone
                print(f"\n[1/5] Creating image tiles for {zone_name}...")
                tile_image(
                    gdf_path=gpkg_path,
                    layer=layer,
                    tif_paths=bands,
                    output_dir=test_images_dir,
                    file_prefix=f'{reserve}_{zone_name}_',
                    tile_size=tile_size,
                    clear_dir=True
                )
                num_tiles = len([f for f in os.listdir(test_images_dir) if f.endswith('.tif')])
                print(f"  ✓ Created {num_tiles} image tiles")
            
                # Clip labels for test zone
                print(f"\n[2/5] Clipping labels for {zone_name}...")
                clip_geotiffs(
                    label_path,
                    reference_dir=test_images_dir,
                    output_dir=test_labels_dir
                )
                num_labels = len([f for f in os.listdir(test_labels_dir) if f.endswith('.tif')])
                print(f"  ✓ Created {num_labels} label tiles")
                
                # CRITICAL DEBUG: Check if any label tiles contain maire pixels
                print(f"\n  🔍 DEBUGGING: Inspecting label tiles for maire pixels...")
                label_files_debug = [f for f in os.listdir(test_labels_dir) if f.endswith('.tif')]
                tiles_with_maire = 0
                total_maire_pixels_in_labels = 0
                for label_file_debug in label_files_debug[:10]:  # Check first 10 files
                    label_arr = np.array(Image.open(os.path.join(test_labels_dir, label_file_debug)))
                    unique_vals = np.unique(label_arr)
                    maire_pixels = (label_arr > 0).sum()  # Any non-zero value is maire
                    if maire_pixels > 0:
                        tiles_with_maire += 1
                        total_maire_pixels_in_labels += maire_pixels
                        print(f"    {label_file_debug}: {maire_pixels:,} maire pixels (unique: {unique_vals})")
                if tiles_with_maire == 0:
                    print(f"  ⚠️  WARNING: NO MAIRE PIXELS found in first {min(10, len(label_files_debug))} label tiles!")
                    print(f"    This means the test zone may not overlap with maire labels")
                else:
                    print(f"  ✓ Found maire pixels in {tiles_with_maire}/{min(10, len(label_files_debug))} label tiles")
                    print(f"    Total maire pixels in sampled tiles: {total_maire_pixels_in_labels:,}")
            
                # Run predictions
                print(f"\n[3/5] Running inference for {zone_name}...")
                print(f"  Using model: {model_name}")
                make_predictions(test_images_dir, model_path, test_preds_dir, output_mode=output_mode)
                num_preds = len([f for f in os.listdir(test_preds_dir) if f.endswith('.tif')])
                print(f"  ✓ Generated {num_preds} prediction tiles")
            
                # Copy metadata and merge predictions into single raster
                print(f"\n[4/5] Stitching predictions for {zone_name}...")
                print(f"  Copying metadata from image tiles to prediction tiles...")
                copy_metadata(rgb_dir=test_images_dir, gray_dir=test_preds_dir)
                print(f"  ✓ Metadata copied")
                
                # Save stitched prediction with descriptive name using actual layer name
                layer_name = zone_config['layer']  # Get actual layer name from config
                stitched_pred_path = os.path.join(predictions_out_dir, f'{model_name}_{layer_name}.tif')
                print(f"  Merging {num_preds} tiles into single GeoTIFF...")
                merge_geotiffs(test_preds_dir, stitched_pred_path)
                print(f"  ✓ Stitched prediction saved: {os.path.basename(stitched_pred_path)}")
                print(f"    Full path: {stitched_pred_path}")
            
                # Calculate metrics
                print(f"\n[5/5] Calculating metrics for {zone_name}...")
                pred_files = sorted([f for f in os.listdir(test_preds_dir) if f.endswith('.tif')])
                label_files = sorted([f for f in os.listdir(test_labels_dir) if f.endswith('.tif')])
                print(f"  Found {len(pred_files)} prediction files and {len(label_files)} label files")
                
                # Check if files match
                if len(pred_files) != len(label_files):
                    print(f"  WARNING: Mismatch in file counts!")
                    print(f"  Prediction files: {pred_files[:3]}...")
                    print(f"  Label files: {label_files[:3]}...")
                
                print(f"  Processing {len(pred_files)} prediction-label pairs...")
                
                # Initialize metric accumulators
                total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
                iou_eval = IOUEval(len(ALL_CLASSES))
                
                # Debug counters
                total_pred_maire_pixels = 0
                total_label_maire_pixels = 0
            
                for pred_file, label_file in zip(pred_files, label_files):
                    # Check if filenames match (base names should be the same)
                    if pred_file != label_file:
                        print(f"  WARNING: Filename mismatch - pred: {pred_file}, label: {label_file}")
                    
                    # Read prediction and label
                    pred = np.array(Image.open(os.path.join(test_preds_dir, pred_file)))
                    label = np.array(Image.open(os.path.join(test_labels_dir, label_file)))
                    
                    # Ensure same shape
                    if pred.shape != label.shape:
                        print(f"  Warning: Shape mismatch for {pred_file}: {pred.shape} vs {label.shape}")
                        continue
                    
                    # Flatten arrays
                    pred_flat = pred.flatten()
                    label_flat = label.flatten()
                    
                    # Debug: check unique values in first file to diagnose encoding issue
                    if total_pred_maire_pixels == 0 and total_label_maire_pixels == 0:
                        print(f"  Debug - First file ({pred_file}):")
                        print(f"    Prediction unique values: {np.unique(pred_flat)}")
                        print(f"    Label unique values: {np.unique(label_flat)}")
                    
                    # CRITICAL FIX: Normalize to 0/1 if values are 0/255
                    # Predictions are often saved as 0/255, labels might be 0/1 or 0/255
                    if pred_flat.max() > 1:
                        pred_flat = (pred_flat > 127).astype(np.uint8)  # Threshold at midpoint
                    if label_flat.max() > 1:
                        label_flat = (label_flat > 127).astype(np.uint8)
                    
                    # Calculate confusion matrix for maire class (class 1)
                    # Debug: count maire pixels
                    total_pred_maire_pixels += (pred_flat == 1).sum()
                    total_label_maire_pixels += (label_flat == 1).sum()
                    
                    total_tp += ((pred_flat == 1) & (label_flat == 1)).sum()
                    total_fp += ((pred_flat == 1) & (label_flat == 0)).sum()
                    total_fn += ((pred_flat == 0) & (label_flat == 1)).sum()
                    total_tn += ((pred_flat == 0) & (label_flat == 0)).sum()
                    
                    # Update IoU evaluator (convert numpy arrays to torch tensors)
                    pred_tensor = torch.from_numpy(pred_flat)
                    label_tensor = torch.from_numpy(label_flat)
                    iou_eval.addBatch(pred_tensor, label_tensor)
                
                # Print debug info
                print(f"\n  Debug - Pixel counts:")
                print(f"    Total predicted maire pixels: {total_pred_maire_pixels:,}")
                print(f"    Total ground truth maire pixels: {total_label_maire_pixels:,}")
                print(f"    TP={total_tp:,}, FP={total_fp:,}, FN={total_fn:,}, TN={total_tn:,}")
            
                # Calculate metrics
                maire_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                maire_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                maire_f1 = 2 * maire_precision * maire_recall / (maire_precision + maire_recall) if (maire_precision + maire_recall) > 0 else 0.0
                maire_iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
                
                overall_acc, per_class_acc, per_class_iu, miou = iou_eval.getMetric()
                
                # Store results
                test_results[model_name][zone_name] = {
                    'miou': miou,
                    'maire_precision': maire_precision,
                    'maire_recall': maire_recall,
                    'maire_f1': maire_f1,
                    'maire_iou': maire_iou
                }
            
                # Print results
                print(f"\n  ✓ Metrics calculated successfully:")
                print(f"    Test Zone:       {zone_name}")
                print(f"    Model:           {model_name}")
                print(f"    mIoU:            {miou:.4f}")
                print(f"    Maire Precision: {maire_precision:.4f}")
                print(f"    Maire Recall:    {maire_recall:.4f}")
                print(f"    Maire F1:        {maire_f1:.4f}")
                print(f"    Maire IoU:       {maire_iou:.4f}")
                
                # Append to CSV
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([model_name, zone_name, miou, maire_precision, maire_recall, maire_f1, maire_iou])
                print(f"  ✓ Metrics appended to CSV: {os.path.basename(csv_file)}")
            
                # Clean up temporary files
                print(f"\n  Cleaning up temporary directory: {temp_test_dir}")
                try:
                    if os.path.exists(temp_test_dir):
                        shutil.rmtree(temp_test_dir)
                        print(f"  ✓ Successfully removed temporary directory")
                    else:
                        print(f"  ⚠ Temporary directory not found (may have been removed already)")
                except Exception as e:
                    print(f"  ❌ Error cleaning up temporary directory: {e}")
                    print(f"     Directory may need manual cleanup: {temp_test_dir}")
                
                print(f"\n✅ COMPLETED Zone {zone_idx}/{len(test_zones_config)}: {zone_name}")
                print(f"{'='*80}")
                
            except Exception as zone_error:
                print(f"\n💥 CRITICAL ERROR in Zone {zone_idx}/{len(test_zones_config)} ({zone_config['zone_name']}):")
                print(f"   Error: {zone_error}")
                import traceback
                traceback.print_exc()
                print(f"\n⚠️  Skipping to next zone...\n")
                continue
        
        print(f"\n🎯 Completed all {len(test_zones_config)} zones for model {model_name}")
    
    print(f"\n{'='*80}")
    print(f"TEST SET EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Summary:")
    print(f"  Models evaluated: {len([k for k in test_results.keys()])}")
    print(f"  Test zones processed: {len(test_zones_config)}")
    print(f"  Total predictions generated: {len(test_results) * len(test_zones_config)}")
    print(f"  Metrics CSV: {csv_file}")
    print(f"  Stitched predictions: {predictions_out_dir}")
    print(f"{'='*80}\n")
    
    return test_results


def evaluate_stitched_predictions(
    prediction_tif,
    label_gpkg,
    label_layer,
    zone_gpkg,
    zone_layer,
    reference_raster,
    model_name='model',
    zone_name='zone',
    output_csv=None,
    temp_dir=None
):
    """
    Calculate metrics from a stitched prediction GeoTIFF against vector ground truth.
    
    This function is completely independent of the training pipeline and works with:
    - A stitched prediction raster (output from inference)
    - Vector ground truth labels (from GeoPackage)
    - A zone polygon to define the evaluation area
    
    The ground truth is rasterized on-the-fly to match the prediction extent.
    
    Args:
        prediction_tif: Path to the stitched prediction GeoTIFF (binary: 0=background, 255=maire)
        label_gpkg: Path to GeoPackage containing ground truth polygons
        label_layer: Layer name in the label GeoPackage
        zone_gpkg: Path to GeoPackage containing zone polygon
        zone_layer: Layer name for the evaluation zone (e.g., 'unet_test_zone', 'bbox')
        reference_raster: Path to a reference raster for CRS and resolution (usually one of the input bands)
        model_name: Name identifier for the model (for CSV output)
        zone_name: Name identifier for the zone (for CSV output)
        output_csv: Optional path to CSV file to append results
        temp_dir: Optional temporary directory for intermediate files (cleaned up after)
        
    Returns:
        Dictionary with metrics: mIoU, precision, recall, f1, iou for maire class
    """
    import rasterio
    from rasterio.mask import mask
    from rasterio.features import rasterize
    import geopandas as gpd
    from shapely.geometry import box
    import tempfile
    import shutil
    
    print(f"\n{'='*80}")
    print(f"EVALUATING STITCHED PREDICTION")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Zone: {zone_name} (layer: {zone_layer})")
    print(f"Prediction: {os.path.basename(prediction_tif)}")
    print(f"Ground truth: {os.path.basename(label_gpkg)} / {label_layer}")
    print(f"{'='*80}")
    
    # Create temp directory if not provided
    cleanup_temp = False
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix='eval_stitched_')
        cleanup_temp = True
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 1. Load zone polygon
        print(f"\n[1/4] Loading zone polygon...")
        zone_gdf = gpd.read_file(zone_gpkg, layer=zone_layer)
        zone_geom = zone_gdf.geometry.unary_union
        print(f"  ✓ Zone loaded: {zone_layer}")
        
        # 2. Load and clip prediction to zone
        print(f"\n[2/4] Loading and clipping prediction to zone...")
        with rasterio.open(prediction_tif) as pred_src:
            # Reproject zone to prediction CRS if needed
            if zone_gdf.crs != pred_src.crs:
                zone_gdf = zone_gdf.to_crs(pred_src.crs)
                zone_geom = zone_gdf.geometry.unary_union
            
            # Mask prediction to zone
            pred_clipped, pred_transform = mask(pred_src, [zone_geom], crop=True, filled=True, nodata=0)
            pred_array = pred_clipped[0]  # First band
            pred_meta = pred_src.meta.copy()
            pred_meta.update({
                'height': pred_array.shape[0],
                'width': pred_array.shape[1],
                'transform': pred_transform
            })
        
        # Normalize prediction to 0/1
        if pred_array.max() > 1:
            pred_array = (pred_array > 127).astype(np.uint8)
        
        print(f"  ✓ Prediction clipped: {pred_array.shape}")
        print(f"    Unique values: {np.unique(pred_array)}")
        print(f"    Predicted maire pixels: {(pred_array == 1).sum():,}")
        
        # 3. Load and rasterize ground truth labels within zone
        print(f"\n[3/4] Rasterizing ground truth labels...")
        label_gdf = gpd.read_file(label_gpkg, layer=label_layer)
        
        # Reproject labels to prediction CRS if needed
        if label_gdf.crs != rasterio.CRS.from_string(str(pred_meta['crs'])):
            label_gdf = label_gdf.to_crs(pred_meta['crs'])
        
        # Clip labels to zone
        label_gdf = label_gdf.clip(zone_gdf)
        
        # Create label raster matching prediction dimensions
        if len(label_gdf) > 0 and not label_gdf.is_empty.all():
            # Rasterize label polygons
            label_array = rasterize(
                [(geom, 1) for geom in label_gdf.geometry if geom is not None and not geom.is_empty],
                out_shape=pred_array.shape,
                transform=pred_transform,
                fill=0,
                dtype=np.uint8
            )
        else:
            print(f"  ⚠ No label polygons found in zone!")
            label_array = np.zeros_like(pred_array)
        
        print(f"  ✓ Labels rasterized: {label_array.shape}")
        print(f"    Unique values: {np.unique(label_array)}")
        print(f"    Ground truth maire pixels: {(label_array == 1).sum():,}")
        
        # 4. Calculate metrics
        print(f"\n[4/4] Calculating metrics...")
        pred_flat = pred_array.flatten()
        label_flat = label_array.flatten()
        
        # Confusion matrix
        TP = ((pred_flat == 1) & (label_flat == 1)).sum()
        FP = ((pred_flat == 1) & (label_flat == 0)).sum()
        FN = ((pred_flat == 0) & (label_flat == 1)).sum()
        TN = ((pred_flat == 0) & (label_flat == 0)).sum()
        
        # Metrics
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
        
        # mIoU (mean of background IoU and maire IoU)
        bg_iou = TN / (TN + FP + FN) if (TN + FP + FN) > 0 else 0.0
        miou = (bg_iou + iou) / 2
        
        results = {
            'model': model_name,
            'zone': zone_name,
            'miou': float(miou),
            'maire_precision': float(precision),
            'maire_recall': float(recall),
            'maire_f1': float(f1),
            'maire_iou': float(iou),
            'TP': int(TP),
            'FP': int(FP),
            'FN': int(FN),
            'TN': int(TN)
        }
        
        print(f"\n  ✓ Metrics calculated:")
        print(f"    mIoU:            {miou:.4f}")
        print(f"    Maire Precision: {precision:.4f}")
        print(f"    Maire Recall:    {recall:.4f}")
        print(f"    Maire F1:        {f1:.4f}")
        print(f"    Maire IoU:       {iou:.4f}")
        print(f"    Confusion: TP={TP:,}, FP={FP:,}, FN={FN:,}, TN={TN:,}")
        
        # Write to CSV if provided
        if output_csv:
            write_header = not os.path.exists(output_csv)
            with open(output_csv, mode='a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['Model', 'Test Zone', 'mIoU', 'Maire Precision', 'Maire Recall', 'Maire F1', 'Maire IoU', 'TP', 'FP', 'FN', 'TN'])
                writer.writerow([model_name, zone_name, miou, precision, recall, f1, iou, TP, FP, FN, TN])
            print(f"\n  ✓ Results appended to: {output_csv}")
        
        print(f"\n{'='*80}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*80}\n")
        
        return results
        
    finally:
        # Cleanup temp directory
        if cleanup_temp and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def evaluate_all_zones(
    model_output_dir,
    reserve,
    label_gpkg,
    label_layer,
    zone_gpkg,
    zone_layers=['unet_test_zone', 'bbox'],
    reference_raster=None,
    model_types=['best_model_maire_f1', 'best_model_iou']
):
    """
    Evaluate all zone/model combinations for a trained model.
    
    This is a convenience wrapper that finds stitched predictions and evaluates them.
    
    Args:
        model_output_dir: Directory containing model outputs and test_predictions/
        reserve: Reserve name (e.g., 'ESK')
        label_gpkg: Path to GeoPackage with ground truth polygons
        label_layer: Layer name in label GeoPackage
        zone_gpkg: Path to GeoPackage with zone polygons
        zone_layers: List of zone layers to evaluate
        reference_raster: Path to reference raster (for CRS/resolution)
        model_types: List of model checkpoint names to evaluate
        
    Returns:
        Dictionary with all results
    """
    predictions_dir = os.path.join(model_output_dir, 'test_predictions')
    output_csv = os.path.join(model_output_dir, 'test_metrics_stitched.csv')
    
    # Clear existing CSV
    if os.path.exists(output_csv):
        os.remove(output_csv)
    
    print(f"\n{'#'*80}")
    print(f"BATCH EVALUATION: {reserve}")
    print(f"{'#'*80}")
    print(f"Predictions dir: {predictions_dir}")
    print(f"Zone layers: {zone_layers}")
    print(f"Model types: {model_types}")
    print(f"Output CSV: {output_csv}")
    print(f"{'#'*80}\n")
    
    all_results = {}
    
    for model_type in model_types:
        model_name = model_type.replace('best_model_', 'best_')
        all_results[model_name] = {}
        
        for zone_layer in zone_layers:
            # Find the prediction file
            pred_filename = f"{model_type}_{zone_layer}.tif"
            pred_path = os.path.join(predictions_dir, pred_filename)
            
            if not os.path.exists(pred_path):
                print(f"⚠ Prediction not found: {pred_filename}")
                continue
            
            zone_name = f"{reserve}_{zone_layer.replace('unet_', '').replace('_zone', '')}"
            
            result = evaluate_stitched_predictions(
                prediction_tif=pred_path,
                label_gpkg=label_gpkg,
                label_layer=label_layer,
                zone_gpkg=zone_gpkg,
                zone_layer=zone_layer,
                reference_raster=reference_raster,
                model_name=model_name,
                zone_name=zone_name,
                output_csv=output_csv
            )
            
            all_results[model_name][zone_name] = result
    
    print(f"\n{'#'*80}")
    print(f"BATCH EVALUATION COMPLETE")
    print(f"Results saved to: {output_csv}")
    print(f"{'#'*80}\n")
    
    return all_results