"""
U-Net model configuration settings.
Defines device, class labels, weights, and color mappings for segmentation.
"""

import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MAIRE_WEIGHT = 10.0

ALL_CLASSES = ['background', 'swamp_maire']

CLASS_WEIGHTS = torch.tensor([1.0, MAIRE_WEIGHT]).to(DEVICE)

LABEL_COLORS_LIST = [
    (0, 0, 0), # Background.
    (255, 255, 255),
]

VIS_LABEL_MAP = [
    (0, 0, 0), # Background.
    (0, 255, 0),
]