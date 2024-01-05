import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms


def crop_center(matrix, target_size):
    # Calculate matrix center coordinates
    center_x, center_y = matrix.shape[1] // 2, matrix.shape[0] // 2

    # Calculate coordinates for cropping
    crop_x = center_x - target_size[1] // 2
    crop_y = center_y - target_size[0] // 2
    crop_x_end = crop_x + target_size[1]
    crop_y_end = crop_y + target_size[0]

    # Crop the center part
    cropped_matrix = matrix[crop_y:crop_y_end, crop_x:crop_x_end]

    return cropped_matrix

class Halftone(nn.Module):
    """
    对编码图像进行半色调处理
    """
    def __init__(self):
        super(Halftone, self).__init__()
        # 读取半色调矩阵
        self.halftone_matrix = torch.from_numpy(np.load("noise_layers/test_data/screen_matrx.npy")).float()
        self.normalize_transform = transforms.Normalize([0.5], [0.5])


    def forward(self, encoded_image):
        # 裁剪半色调矩阵
        cropped_halftone = crop_center(self.halftone_matrix, encoded_image.shape[2:4])
        cropped_halftone = cropped_halftone.unsqueeze(0).unsqueeze(0)
        normalized_threshold_matrix = self.normalize_transform(cropped_halftone)
        # Apply halftone
        noisy_image = torch.where(encoded_image >= normalized_threshold_matrix, 1.0, -1.0)
        return noisy_image