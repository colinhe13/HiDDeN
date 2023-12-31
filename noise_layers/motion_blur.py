import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np




class MotionBlur(nn.Module):
    def __init__(self, kernel_size=15, angle=30):
        super(MotionBlur, self).__init__()
        self.kernel_size = kernel_size
        self.angle = angle

        # Create a motion blur kernel
        # kernel = np.zeros((kernel_size, kernel_size))
        # kernel[int((kernel_size - 1) / 2), :] = 1.0
        # kernel /= kernel_size
        # kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        # self.register_buffer('weight', kernel.expand(3, -1, -1, -1))

        # # 创建一个运动模糊卷积核
        # kernel = torch.zeros((kernel_size, kernel_size))
        # direction = torch.tensor([1.0, 0.0])
        # center = (kernel_size - 1) / 2
        #
        # for i in range(kernel_size):
        #     offset = i - center
        #     kernel[i, :] = direction * offset
        #
        # kernel /= kernel_size
        # kernel = kernel.view(1, 1, kernel_size, kernel_size)
        # self.register_buffer('weight', kernel)


    def forward(self, x):

        # Create a motion blur kernel
        kernel = torch.zeros((3, 1, self.kernel_size, self.kernel_size), dtype=torch.float32)
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                if i == j:
                    kernel[:, :, i, j] = 1.0
        kernel /= self.kernel_size

        # Apply the motion blur kernel
        # x = F.conv2d(x, self.weight, padding=self.kernel_size // 2, groups=3)
        x = F.conv2d(x, kernel, padding=0, groups=3)
        return x