import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np




# class MotionBlur(nn.Module):
#     def __init__(self, kernel_size=15, angle=30):
#         super(MotionBlur, self).__init__()
#         self.kernel_size = kernel_size
#         self.angle = angle
#
#     def forward(self, x):
#
#         # Create a motion blur kernel
#         kernel = torch.zeros((3, 1, self.kernel_size, self.kernel_size), dtype=torch.float32)
#         for i in range(self.kernel_size):
#             for j in range(self.kernel_size):
#                 if i == j:
#                     kernel[:, :, i, j] = 1.0
#         kernel /= self.kernel_size
#
#         # Apply the motion blur kernel
#         x = F.conv2d(x, kernel, padding=0, groups=3)
#         return x


import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionBlur(nn.Module):
    def __init__(self, blur_range=(5, 10), angle_range=(0, 360)):
        super(MotionBlur, self).__init__()
        self.blur_range = blur_range
        self.angle_range = angle_range

    def forward(self, noised_and_cover):
        # Generate random blur size and angle
        blur_size = torch.randint(self.blur_range[0], self.blur_range[1] + 1, (1,)).item()
        blur_angle = torch.randint(self.angle_range[0], self.angle_range[1] + 1, (1,)).item()
        # print(blur_size, blur_angle)

        # Create a motion blur kernel
        kernel = self._motion_blur_kernel(blur_size, blur_angle)

        # Apply convolution to each channel
        blurred_image = F.conv2d(noised_and_cover[0], kernel, padding=0, groups=3)

        noised_and_cover[0] = blurred_image

        return noised_and_cover

    def _motion_blur_kernel(self, size, angle):
        kernel = torch.zeros((1, 1, size, size), dtype=torch.float32)

        # Convert angle to radians
        angle_rad = angle * (torch.pi / 180.0)
        angle_rad = torch.tensor(angle_rad)

        # Calculate kernel coordinates based on angle
        x = torch.cos(angle_rad)
        y = torch.sin(angle_rad)

        # Set the line in the middle of the kernel
        line = torch.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, steps=size)
        line = x * line
        line = torch.round(line + (size - 1) / 2.0).long()

        # Set kernel values along the line
        kernel[0, 0, line, line] = 1.0

        # Normalize the kernel
        kernel /= kernel.sum()

        # Expand to 3 channels
        kernel = kernel.expand(3, 1, -1, -1)

        return kernel
