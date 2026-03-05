# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# def double_convolution(in_channels, out_channels):
#     """
#     In the original paper implementation, the convolution operations were
#     not padded but we are padding them here. This is because, we need the 
#     output result size to be same as input size.
#     """
#     conv_op = nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(inplace=True)
#     )
#     return conv_op

# class UNet(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(UNet, self).__init__()

#         self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

#         # Contracting path.
#         # Each convolution is applied twice.
#         self.down_convolution_1 = double_convolution(in_channels, 64)
#         self.down_convolution_2 = double_convolution(64, 128)
#         self.down_convolution_3 = double_convolution(128, 256)
#         self.down_convolution_4 = double_convolution(256, 512)
#         self.down_convolution_5 = double_convolution(512, 1024)

#         # Expanding path.
#         self.up_transpose_1 = nn.ConvTranspose2d(
#             in_channels=1024, 
#             out_channels=512,
#             kernel_size=2, 
#             stride=2)
#         # Below, `in_channels` again becomes 1024 as we are concatinating.
#         self.up_convolution_1 = double_convolution(1024, 512)
#         self.up_transpose_2 = nn.ConvTranspose2d(
#             in_channels=512, 
#             out_channels=256,
#             kernel_size=2, 
#             stride=2)
#         self.up_convolution_2 = double_convolution(512, 256)
#         self.up_transpose_3 = nn.ConvTranspose2d(
#             in_channels=256, 
#             out_channels=128,
#             kernel_size=2, 
#             stride=2)
#         self.up_convolution_3 = double_convolution(256, 128)
#         self.up_transpose_4 = nn.ConvTranspose2d(
#             in_channels=128, 
#             out_channels=64,
#             kernel_size=2, 
#             stride=2)
#         self.up_convolution_4 = double_convolution(128, 64)

#         # output => increase the `out_channels` as per the number of classes.
#         self.out = nn.Conv2d(
#             in_channels=64, 
#             out_channels=num_classes, 
#             kernel_size=1
#         ) 

    # def forward(self, x):
    #     down_1 = self.down_convolution_1(x)
    #     down_2 = self.max_pool2d(down_1)
    #     down_3 = self.down_convolution_2(down_2)
    #     down_4 = self.max_pool2d(down_3)
    #     down_5 = self.down_convolution_3(down_4)
    #     down_6 = self.max_pool2d(down_5)
    #     down_7 = self.down_convolution_4(down_6)
    #     down_8 = self.max_pool2d(down_7)
    #     down_9 = self.down_convolution_5(down_8)        
    #     # *** DO NOT APPLY MAX POOL TO down_9 ***
        
    #     up_1 = self.up_transpose_1(down_9)
    #     x = self.up_convolution_1(torch.cat([down_7, up_1], 1))

    #     up_2 = self.up_transpose_2(x)
    #     x = self.up_convolution_2(torch.cat([down_5, up_2], 1))

    #     up_3 = self.up_transpose_3(x)
    #     x = self.up_convolution_3(torch.cat([down_3, up_3], 1))

    #     up_4 = self.up_transpose_4(x)
    #     x = self.up_convolution_4(torch.cat([down_1, up_4], 1))

    #     out = self.out(x)
    #     return out


import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Configure logging
logging.basicConfig(
    filename="unet_shapes.log",  # Log file name
    level=logging.INFO,          # Log level
    format="%(asctime)s - %(message)s"
)

def log_shape(name, tensor):
    logging.info(f"{name}: {tensor.shape}")

def double_convolution(in_channels, out_channels):
    """
    Two Conv-BN-ReLU blocks with padding to preserve spatial size.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, output_mode='multiclass'):
        """
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            output_mode: 'multiclass' (default) or 'binary'
                - 'multiclass': outputs num_classes channels (for CrossEntropyLoss)
                - 'binary': outputs 1 channel with sigmoid (for BCEWithLogitsLoss/BCELoss)
        """
        super(UNet, self).__init__()
        self.output_mode = output_mode

        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        # Contracting path
        self.down_convolution_1 = double_convolution(in_channels, 64)
        self.down_convolution_2 = double_convolution(64, 128)
        self.down_convolution_3 = double_convolution(128, 256)
        self.down_convolution_4 = double_convolution(256, 512)
        self.down_convolution_5 = double_convolution(512, 1024)

        # Expanding path
        self.up_transpose_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_convolution_1 = double_convolution(1024, 512)

        self.up_transpose_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_convolution_2 = double_convolution(512, 256)

        self.up_transpose_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_convolution_3 = double_convolution(256, 128)

        self.up_transpose_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_convolution_4 = double_convolution(128, 64)

        # Output layer
        if output_mode == 'binary':
            self.out = nn.Conv2d(64, 1, kernel_size=1)  # Single channel for binary
        else:
            self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        log_shape("Input", x)

        # Contracting path
        down_1 = self.down_convolution_1(x)
        log_shape("After down_convolution_1", down_1)

        down_2 = self.max_pool2d(down_1)
        log_shape("After max_pool (1)", down_2)

        down_3 = self.down_convolution_2(down_2)
        log_shape("After down_convolution_2", down_3)

        down_4 = self.max_pool2d(down_3)
        log_shape("After max_pool (2)", down_4)

        down_5 = self.down_convolution_3(down_4)
        log_shape("After down_convolution_3", down_5)

        down_6 = self.max_pool2d(down_5)
        log_shape("After max_pool (3)", down_6)

        down_7 = self.down_convolution_4(down_6)
        log_shape("After down_convolution_4", down_7)

        down_8 = self.max_pool2d(down_7)
        log_shape("After max_pool (4)", down_8)

        down_9 = self.down_convolution_5(down_8)
        log_shape("After down_convolution_5 (bottom)", down_9)

        # Expanding path
        up_1 = self.up_transpose_1(down_9)
        log_shape("After up_transpose_1", up_1)
        x = self.up_convolution_1(torch.cat([down_7, up_1], 1))
        log_shape("After up_convolution_1", x)

        up_2 = self.up_transpose_2(x)
        log_shape("After up_transpose_2", up_2)
        x = self.up_convolution_2(torch.cat([down_5, up_2], 1))
        log_shape("After up_convolution_2", x)

        up_3 = self.up_transpose_3(x)
        log_shape("After up_transpose_3", up_3)
        x = self.up_convolution_3(torch.cat([down_3, up_3], 1))
        log_shape("After up_convolution_3", x)

        up_4 = self.up_transpose_4(x)
        log_shape("After up_transpose_4", up_4)
        x = self.up_convolution_4(torch.cat([down_1, up_4], 1))
        log_shape("After up_convolution_4", x)

        out = self.out(x)
        log_shape("Output", out)
        return out