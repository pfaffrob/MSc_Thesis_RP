import glob
import numpy as np
import torch
import albumentations as A
import cv2
import rasterio
from .utils import get_label_mask, set_class_values
from torch.utils.data import Dataset, DataLoader

def get_images(root_path):
    train_images = glob.glob(f"{root_path}/train_images/*")
    train_images.sort()
    train_masks = glob.glob(f"{root_path}/train_masks/*")
    train_masks.sort()
    valid_images = glob.glob(f"{root_path}/valid_images/*")
    valid_images.sort()
    valid_masks = glob.glob(f"{root_path}/valid_masks/*")
    valid_masks.sort()

    return train_images, train_masks, valid_images, valid_masks

def train_transforms(img_size=None):
    """
    Transforms/augmentations for training images and masks.
    
    All augmentations are n-channel compatible (work with any number of bands).
    Optimized for maire detection with streamlined augmentation strategy.
    
    :param img_size: Integer, for image resize. If None, no resizing is applied.
    """
    transforms_list = [
        # Basic geometric augmentations (n-channel safe)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        # Combined geometric transformation (includes rotation + scale + shift)
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.5),
        
        # Mild grid distortion - handles terrain/canopy variations
        A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.2),
        
        # Subtle intensity augmentation - illumination changes
        A.MultiplicativeNoise(multiplier=[0.95, 1.05], p=0.2),
        
        # ❌ RGB-ONLY augmentations (commented out for multispectral compatibility):
        # A.RandomBrightnessContrast(p=0.2)  # Only works with 3 channels
        # A.RandomSunFlare(p=0.2)            # Only works with 3 channels
        # A.RandomFog(p=0.2)                 # Only works with 3 channels
    ]

    # Only add resizing if img_size is provided
    if img_size is not None:
        transforms_list.insert(0, A.Resize(img_size, img_size, always_apply=True))

    train_image_transform = A.Compose(transforms_list)
    
    return train_image_transform

def valid_transforms(img_size=None):
    """
    Transforms/augmentations for validation images and masks.

    :param img_size: Integer, for image resize. If None, no resizing is applied.
    """
    transforms_list = []

    # Only add resizing if img_size is provided
    if img_size is not None:
        transforms_list.append(A.Resize(img_size, img_size, always_apply=True))

    # Ensure we always return a callable transform
    valid_image_transform = A.Compose(transforms_list if transforms_list else [A.NoOp()])

    return valid_image_transform


class SegmentationDataset(Dataset):
    def __init__(
        self, 
        image_paths, 
        mask_paths, 
        tfms, 
        label_colors_list,
        classes_to_train,
        all_classes,
        loss_type='ce'  # 'ce' for CrossEntropy, 'bce' for Binary CrossEntropy
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.tfms = tfms
        self.label_colors_list = label_colors_list
        self.all_classes = all_classes
        self.classes_to_train = classes_to_train
        self.loss_type = loss_type
        # Convert string names to class values for masks.
        self.class_values = set_class_values(
            self.all_classes, self.classes_to_train
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]


        with rasterio.open(image_path) as src:
            image = src.read()
            dtype = image.dtype

            # Normalize based on dtype
            if dtype == np.uint8:
                image = image.astype('float32') / 255.0
            elif dtype == np.uint16:
                image = image.astype('float32') / 65535.0
            else:
                raise ValueError(f"Unsupported image dtype: {dtype}")


        mask_path = self.mask_paths[index]
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB).astype('float32')

        im = mask > 0
        mask[im] = 255
        mask[np.logical_not(im)] = 0

        # Transpose image to (height, width, bands) for albumentations
        image = np.transpose(image, (1, 2, 0))

        transformed = self.tfms(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        mask = get_label_mask(mask, self.class_values, self.label_colors_list)

        # Transpose back to (bands, height, width) for PyTorch
        image = np.transpose(image, (2, 0, 1))

        image = torch.tensor(image, dtype=torch.float)
        
        # For BCE loss, mask needs to be float with shape (1, H, W) for single channel
        # For CE loss, mask is long with shape (H, W) for class indices
        if self.loss_type == 'bce':
            # Convert to binary float mask: class 0 (background) -> 0.0, class 1 (maire) -> 1.0
            mask = torch.tensor(mask, dtype=torch.float).unsqueeze(0)  # Add channel dimension
        else:
            mask = torch.tensor(mask, dtype=torch.long)

        return image, mask


def get_dataset(
    train_image_paths, 
    train_mask_paths,
    valid_image_paths,
    valid_mask_paths,
    all_classes,
    classes_to_train,
    label_colors_list,
    img_size = None,
    loss_type='ce'  # 'ce' for CrossEntropy, 'bce' for Binary CrossEntropy
):
    train_tfms = train_transforms(img_size)
    valid_tfms = valid_transforms(img_size)

    train_dataset = SegmentationDataset(
        train_image_paths,
        train_mask_paths,
        train_tfms,
        label_colors_list,
        classes_to_train,
        all_classes,
        loss_type=loss_type
    )
    valid_dataset = SegmentationDataset(
        valid_image_paths,
        valid_mask_paths,
        valid_tfms,
        label_colors_list,
        classes_to_train,
        all_classes,
        loss_type=loss_type
    )
    return train_dataset, valid_dataset

def get_data_loaders(train_dataset, valid_dataset, batch_size):
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, drop_last=False
    )
    valid_data_loader = DataLoader(
        valid_dataset, batch_size=batch_size, drop_last=False
    )

    return train_data_loader, valid_data_loader