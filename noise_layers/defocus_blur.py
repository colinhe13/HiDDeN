import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF



# class DefocusBlur(nn.Module):
#     def __init__(self, kernel_size):
#         super(DefocusBlur, self).__init__()
#         self.kernel_size = kernel_size
#
#     def forward(self, x):
#
#         # 创建深度通道数为3的卷积核
#         kernel = torch.ones((3, 1, self.kernel_size, self.kernel_size), dtype=torch.float32) / (self.kernel_size ** 2)
#         # 对每个通道应用卷积操作
#         blurred_image = F.conv2d(x, kernel, padding=0, groups=3)
#
#         return blurred_image

class DefocusBlur(nn.Module):
    def __init__(self, blur_range=(5, 10)):
        super(DefocusBlur, self).__init__()
        self.blur_range = blur_range

    def forward(self, noised_and_cover):
        # Generate a random blur size
        blur_size = torch.randint(self.blur_range[0], self.blur_range[1] + 1, (1,)).item()

        # Create a depth channel (1 channel) convolution kernel
        kernel = torch.ones((3, 1, blur_size, blur_size), dtype=torch.float32) / (blur_size ** 2)

        # Apply convolution to each channel
        blurred_image = F.conv2d(noised_and_cover[0], kernel, padding=0, groups=3)

        noised_and_cover[0] = blurred_image

        return noised_and_cover