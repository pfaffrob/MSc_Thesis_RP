import numpy as np
import cv2
import torch
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import csv
from .model import UNet
from .config import ALL_CLASSES, DEVICE

from torchvision import transforms
from .config import (
    VIS_LABEL_MAP as viz_map
)

plt.style.use('ggplot')

def set_class_values(all_classes, classes_to_train):
    """
    This (`class_values`) assigns a specific class label to the each of the classes.
    For example, `animal=0`, `archway=1`, and so on.

    :param all_classes: List containing all class names.
    :param classes_to_train: List containing class names to train.
    """
    class_values = [all_classes.index(cls.lower()) for cls in classes_to_train]
    return class_values

def get_label_mask(mask, class_values, label_colors_list):
    """
    This function encodes the pixels belonging to the same class
    in the image into the same label

    :param mask: NumPy array, segmentation mask.
    :param class_values: List containing class values, e.g car=0, bus=1.
    :param label_colors_list: List containing RGB color value for each class.
    """
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for value in class_values:
        for ii, label in enumerate(label_colors_list):
            if value == label_colors_list.index(label):
                label = np.array(label)
                label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = value
    label_mask = label_mask.astype(int)
    return label_mask

def draw_translucent_seg_maps(
    data, 
    output, 
    epoch, 
    i, 
    val_seg_dir, 
    label_colors_list,
):
    """
    This function color codes the segmentation maps that is generated while
    validating. THIS IS NOT TO BE CALLED FOR SINGLE IMAGE TESTING
    """
    alpha = 1 # how much transparency
    beta = 0.8 # alpha + beta should be 1
    gamma = 0 # contrast

    seg_map = output[0] # use only one output from the batch
    
    # Handle both BCE (single channel) and CE (multi-channel) outputs
    if seg_map.shape[0] == 1:  # BCE mode: single channel output
        # Apply sigmoid and threshold at 0.5 to get binary mask
        seg_map = torch.sigmoid(seg_map.squeeze()).detach().cpu().numpy()
        seg_map = (seg_map > 0.5).astype(np.uint8)  # 0 for background, 1 for maire
    else:  # CE mode: multi-channel output
        seg_map = torch.argmax(seg_map.squeeze(), dim=0).detach().cpu().numpy()

    image = data[0][:3]
    image = np.array(image.cpu())
    image = np.transpose(image, (1, 2, 0))
    # unnormalize the image (important step)
    # mean = np.array([0.5, 0.5, 0.5])
    # std = np.array([0.5, 0.5, 0.5])
    # image = std * image + mean
    image = np.array(image, dtype=np.float32)
    image = image * 255

    red_map = np.zeros_like(seg_map).astype(np.uint8)
    green_map = np.zeros_like(seg_map).astype(np.uint8)
    blue_map = np.zeros_like(seg_map).astype(np.uint8)


    for label_num in range(0, len(label_colors_list)):
        index = seg_map == label_num
        red_map[index] = np.array(viz_map)[label_num, 0]
        green_map[index] = np.array(viz_map)[label_num, 1]
        blue_map[index] = np.array(viz_map)[label_num, 2]
        
    rgb = np.stack([red_map, green_map, blue_map], axis=2)
    rgb = np.array(rgb, dtype=np.float32)
    # convert color to BGR format for OpenCV
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # cv2.imshow('rgb', rgb)
    # cv2.waitKey(0)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(image, alpha, rgb, beta, gamma, image)
    cv2.imwrite(f"{val_seg_dir}/e{epoch}_b{i}.jpg", image)

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    Uses epoch 5 as baseline - only starts saving from epoch 10 onwards.
    """
    def __init__(self, best_valid_loss=float('inf'), baseline_epoch=10):
        self.best_valid_loss = best_valid_loss
        self.baseline_epoch = baseline_epoch
        self.epoch5_loss = None
        
    def __call__(
        self, current_valid_loss, epoch, model, out_dir, name='model'
    ):
        # Store epoch 5 as baseline
        if epoch + 1 == self.baseline_epoch:
            self.epoch5_loss = current_valid_loss
            self.best_valid_loss = current_valid_loss
            print(f"\n[Baseline] Epoch {self.baseline_epoch} validation loss: {current_valid_loss:.4f} (baseline set)")
            print(f"Saving baseline model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                }, os.path.join(out_dir, 'best_'+name+'.pth'))
        # After epoch 5, only save if better than current best
        elif epoch + 1 > self.baseline_epoch and current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss:.4f} (baseline: {self.epoch5_loss:.4f})")
            print(f"Saving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                }, os.path.join(out_dir, 'best_'+name+'.pth'))

class SaveBestModelIOU:
    """
    Class to save the best model while training. If the current epoch's 
    IoU is higher than the previous highest, then save the
    model state.
    Uses epoch 5 as baseline - only starts saving from epoch 10 onwards.
    """
    def __init__(self, best_iou=float('-inf'), baseline_epoch=10):
        self.best_iou = best_iou
        self.baseline_epoch = baseline_epoch
        self.epoch5_iou = None
        
    def __call__(self, current_iou, epoch, model, out_dir, name='model'):
        """Save model when current_iou is strictly greater than stored best.

        This method safely coerces common types (torch.Tensor, numpy array,
        lists, or scalars) to a Python float before comparison. When a new
        best is found the model is saved and the best value is persisted to
        a small text file in out_dir so resumed training can pick it up.
        Uses epoch 5 as baseline - only starts saving from epoch 10 onwards.
        """
        # Normalize current_iou to a float
        try:
            if hasattr(current_iou, 'item'):
                curr = float(current_iou.item())
            else:
                # handles numpy scalars/arrays and python lists
                curr = float(np.array(current_iou).astype(float).tolist())
        except Exception:
            # fallback: attempt direct float conversion
            try:
                curr = float(current_iou)
            except Exception:
                # if we can't convert, skip saving
                print(f"Unable to interpret current_iou={current_iou}; skipping save_best_iou check")
                return

        # Store epoch 10 as baseline
        if epoch + 1 == self.baseline_epoch:
            self.epoch10_iou = curr
            self.best_iou = curr
            print(f"\n[Baseline] Epoch {self.baseline_epoch} validation IoU: {self.best_iou:.4f} (baseline set)")
            print(f"Saving baseline model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                }, os.path.join(out_dir, 'best_'+name+'.pth'))
            
            # Persist the best IoU so resumed training can reuse it
            try:
                with open(os.path.join(out_dir, 'best_iou.txt'), 'w') as f:
                    f.write(str(self.best_iou))
            except Exception:
                pass
        # After epoch 10, only save if better than current best
        elif epoch + 1 > self.baseline_epoch and curr > self.best_iou:
            self.best_iou = curr
            print(f"\nBest validation IoU: {self.best_iou:.4f} (baseline: {self.epoch10_iou:.4f})")
            print(f"Saving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                }, os.path.join(out_dir, 'best_'+name+'.pth'))
            
            # Persist the best IoU so resumed training can reuse it
            try:
                with open(os.path.join(out_dir, 'best_iou.txt'), 'w') as f:
                    f.write(str(self.best_iou))
            except Exception:
                # non-fatal if we cannot persist
                pass

class SaveBestModelMaireF1:
    """
    Class to save the best model while training based on maire F1 score.
    If the current epoch's maire F1 is higher than the previous highest,
    then save the model state.
    Uses epoch 5 as baseline - only starts saving from epoch 5 onwards.
    """
    def __init__(self, best_maire_f1=float('-inf'), baseline_epoch=10):
        self.best_maire_f1 = best_maire_f1
        self.baseline_epoch = baseline_epoch
        self.epoch5_maire_f1 = None
        
    def __call__(self, current_maire_f1, epoch, model, out_dir, name='model'):
        """Save model when current maire F1 is strictly greater than stored best.
        Uses epoch 5 as baseline - only starts saving from epoch 5 onwards.
        """
        # Normalize current_maire_f1 to a float
        try:
            if hasattr(current_maire_f1, 'item'):
                curr = float(current_maire_f1.item())
            else:
                curr = float(np.array(current_maire_f1).astype(float).tolist())
        except Exception:
            try:
                curr = float(current_maire_f1)
            except Exception:
                print(f"Unable to interpret current_maire_f1={current_maire_f1}; skipping save_best_maire_f1 check")
                return

        # Store epoch 5 as baseline
        if epoch + 1 == self.baseline_epoch:
            self.epoch5_maire_f1 = curr
            self.best_maire_f1 = curr
            print(f"\n[Baseline] Epoch {self.baseline_epoch} validation Maire F1: {self.best_maire_f1:.4f} (baseline set)")
            print(f"Saving baseline maire F1 model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                }, os.path.join(out_dir, 'best_'+name+'.pth'))
            
            # Persist the best maire F1 so resumed training can reuse it
            try:
                with open(os.path.join(out_dir, 'best_maire_f1.txt'), 'w') as f:
                    f.write(str(self.best_maire_f1))
            except Exception:
                pass
        # After epoch 5, only save if better than current best
        elif epoch + 1 > self.baseline_epoch and curr > self.best_maire_f1:
            self.best_maire_f1 = curr
            print(f"\nBest validation Maire F1: {self.best_maire_f1:.4f} (baseline: {self.epoch5_maire_f1:.4f})")
            print(f"Saving best maire F1 model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                }, os.path.join(out_dir, 'best_'+name+'.pth'))
            
            # Persist the best maire F1 so resumed training can reuse it
            try:
                with open(os.path.join(out_dir, 'best_maire_f1.txt'), 'w') as f:
                    f.write(str(self.best_maire_f1))
            except Exception:
                pass

def update_plots(fig, ax1, ax2, ax3, ax4, train_f1, valid_f1, train_loss, valid_loss, train_miou, valid_miou, lr_history, epochs, loss_name='Loss'):
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

    # Define x-axis ticks
    max_ticks = 14
    interval = max(10, (epochs // max_ticks + 9) // 10 * 10)  # Ensure interval is divisible by 10
    tick_positions = list(range(0, epochs + 1, interval))
    
    # Ensure that the last tick is always at the final epoch
    if tick_positions[-1] < epochs:
        tick_positions.append(epochs)

    # F1 Score plot (replacing Accuracy)
    ax1.plot(range(1, len(train_f1) + 1), train_f1, color='tab:blue', linestyle='-', label='train F1')
    ax1.plot(range(1, len(valid_f1) + 1), valid_f1, color='tab:red', linestyle='-', label='validation F1')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('F1 Score')
    ax1.legend()
    ax1.set_xticks(tick_positions)

    # Loss plot (with loss type in title)
    ax2.plot(range(1, len(train_loss) + 1), train_loss, color='tab:blue', linestyle='-', label='train loss')
    ax2.plot(range(1, len(valid_loss) + 1), valid_loss, color='tab:red', linestyle='-', label='validation loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'Loss ({loss_name})')
    ax2.legend()
    ax2.set_xticks(tick_positions)

    # mIOU plot
    ax3.plot(range(1, len(train_miou) + 1), train_miou, color='tab:blue', linestyle='-', label='train mIoU')
    ax3.plot(range(1, len(valid_miou) + 1), valid_miou, color='tab:red', linestyle='-', label='validation mIoU')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('mIOU')
    ax3.set_title('mIOU')
    ax3.legend()
    ax3.set_xticks(tick_positions)

    # Learning rate plot
    ax4.plot(range(1, len(lr_history) + 1), lr_history, color='tab:green', linestyle='-', label='learning rate')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate')
    ax4.legend()
    ax4.set_xticks(tick_positions)

    # Adding labels for specific points at x-axis breaks
    for ax, (train_data, valid_data) in zip([ax1, ax2, ax3], 
                                            [(train_f1, valid_f1), (train_loss, valid_loss), (train_miou, valid_miou)]):
        if len(train_data) > 0 and len(valid_data) > 0:
            for pos in tick_positions:
                if pos == 0:
                    continue  # Skip labeling for tick 0
                if pos - 1 < len(train_data):
                    ax.annotate(f'{train_data[pos - 1]:.2f}', (pos, train_data[pos - 1]), textcoords="offset points", xytext=(0,10), ha='center', color='tab:blue', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
                if pos - 1 < len(valid_data):
                    ax.annotate(f'{valid_data[pos - 1]:.2f}', (pos, valid_data[pos - 1]), textcoords="offset points", xytext=(0,-15), ha='center', color='tab:red', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
            # Add label for epoch 1
            if len(train_data) > 0:
                ax.annotate(f'{train_data[0]:.2f}', (1, train_data[0]), textcoords="offset points", xytext=(0,10), ha='center', color='tab:blue', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
            if len(valid_data) > 0:
                ax.annotate(f'{valid_data[0]:.2f}', (1, valid_data[0]), textcoords="offset points", xytext=(0,-15), ha='center', color='tab:red', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    # Adding labels for learning rate plot
    if len(lr_history) > 0:
        for pos in tick_positions:
            if pos == 0:
                continue  # Skip labeling for tick 0
            if pos - 1 < len(lr_history):
                ax4.annotate(f'{lr_history[pos - 1]:.5f}', (pos, lr_history[pos - 1]), textcoords="offset points", xytext=(0,10), ha='center', color='tab:green', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        # Add label for epoch 1
        if len(lr_history) > 0:
            ax4.annotate(f'{lr_history[0]:.5f}', (1, lr_history[0]), textcoords="offset points", xytext=(0,10), ha='center', color='tab:green', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    # Set x-axis limits
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(0, epochs + 1)

    clear_output(wait=True)
    display(fig)
    plt.pause(0.001)


def save_model(model, optimizer, criterion, out_dir, state, name='model'):
    """
    Function to save the trained model to disk.
    """
    epoch, train_loss, valid_loss, train_pix_acc, valid_pix_acc, train_miou, valid_miou, lr_history = state
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'train_pix_acc': train_pix_acc,
                'valid_pix_acc': valid_pix_acc,
                'train_miou': train_miou,
                'valid_miou': valid_miou,
                'lr_history': lr_history
                }, os.path.join(out_dir, 'last_'+name+'.pth'))

def load_model(checkpoint_path, optimizer):
    """
    Function to load a saved model and training state.
    """
    checkpoint = torch.load(checkpoint_path, weights_only= False)
    model = UNet(num_classes=len(ALL_CLASSES)).to(DEVICE)
    
    # Load model state dict with strict=False
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    criterion = checkpoint['loss']
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    valid_loss = checkpoint['valid_loss']
    train_pix_acc = checkpoint['train_pix_acc']
    valid_pix_acc = checkpoint['valid_pix_acc']
    train_miou = checkpoint['train_miou']
    valid_miou = checkpoint['valid_miou']
    lr_history = checkpoint['lr_history']
    
    state = (epoch, train_loss, valid_loss, train_pix_acc, valid_pix_acc, train_miou, valid_miou, lr_history)
    
    return model, criterion, state, optimizer

def get_segment_labels(image, model, DEVICE):
    image = image.unsqueeze(0).to(DEVICE) # add a batch dimension
    with torch.no_grad():
        outputs = model(image)
    return outputs

def draw_segmentation_map(outputs):
    labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()

    # create Numpy arrays containing zeros
    # later to be used to fill them with respective red, green, and blue pixels
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)
    
    for label_num in range(0, len(viz_map)):
        index = labels == label_num
        red_map[index] = np.array(viz_map)[label_num, 0]
        green_map[index] = np.array(viz_map)[label_num, 1]
        blue_map[index] = np.array(viz_map)[label_num, 2]
        
    segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
    return segmentation_map

def image_overlay(image, segmented_image):
    alpha = 1 # transparency for the original image
    beta = 1.0 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum

    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    return image

def update_csv(csv_file, epoch, train_loss, valid_loss, train_acc, valid_acc, train_precision, valid_precision, train_recall, valid_recall, train_f1, valid_f1, train_maire_precision, valid_maire_precision, train_maire_recall, valid_maire_recall, train_maire_f1, valid_maire_f1, train_maire_iou, valid_maire_iou, train_miou, valid_miou, lr_history, train_per_class_iu, valid_per_class_iu, train_per_class_acc, valid_per_class_acc):
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        row = [epoch + 1, train_loss, valid_loss, train_acc, valid_acc, 
               train_precision, valid_precision, train_recall, valid_recall, 
               train_f1, valid_f1, 
               train_maire_precision, valid_maire_precision,
               train_maire_recall, valid_maire_recall,
               train_maire_f1, valid_maire_f1,
               train_maire_iou, valid_maire_iou,
               train_miou, valid_miou, lr_history[-1]]
        # Add per-class mIOU values
        row.extend(train_per_class_iu.tolist())
        row.extend(valid_per_class_iu.tolist())
        # Add per-class accuracy values
        row.extend(train_per_class_acc.tolist())
        row.extend(valid_per_class_acc.tolist())
        writer.writerow(row)
