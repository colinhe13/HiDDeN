import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=15, sigma=3):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

        # 创建高斯卷积核
        kernel = self._gaussian_kernel(kernel_size, sigma)
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        self.weight = kernel.repeat(3, 1, 1, 1)

    def forward(self, x):
        # 对图像进行高斯模糊
        # x = F.conv2d(x, self.weight, padding=self.kernel_size // 2, groups=3)
        x = F.conv2d(x, self.weight, padding=0, groups=3)
        return x

    def _gaussian_kernel(self, size, sigma):
        """生成高斯卷积核"""
        coords = torch.arange(size).float()
        coords -= size // 2

        # 生成二维高斯核
        gaussian = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gaussian = gaussian / gaussian.sum()

        # 将高斯核转换为二维张量
        gaussian = gaussian.view(1, 1, size, 1)
        # 将高斯核与其转置相乘，以获得二维高斯核
        gaussian = torch.matmul(gaussian, gaussian.transpose(2, 3))

        return gaussian


