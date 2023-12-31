import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF



class DefocusBlur(nn.Module):
    def __init__(self, kernel_size=10):
        super(DefocusBlur, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):

        # 创建深度通道数为3的卷积核
        kernel = torch.ones((3, 1, self.kernel_size, self.kernel_size), dtype=torch.float32) / (self.kernel_size ** 2)

        # 对每个通道应用卷积操作
        # blurred_image = F.conv2d(x, kernel, padding=self.kernel_size // 2, groups=3)
        blurred_image = F.conv2d(x, kernel, padding=0, groups=3)

        return blurred_image