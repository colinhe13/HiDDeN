import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

class ThresholdNoise(nn.Module):
    def __init__(self, threshold_matrix_path):
        super(ThresholdNoise, self).__init__()
        # Load the threshold matrix
        self.threshold_matrix = torch.from_numpy(np.load(threshold_matrix_path)).float()

    def forward(self, encoded_image):
        # Crop the threshold matrix to match the size of the encoded image
        cropped_threshold = self.crop_center(self.threshold_matrix, encoded_image[0].shape[2:4])
        # cropped_threshold = self.crop_center(self.threshold_matrix, encoded_image.shape[2:4])
        cropped_threshold = cropped_threshold.unsqueeze(0).unsqueeze(0)

        # Apply thresholding
        noisy_image = torch.where(encoded_image >= cropped_threshold, 1.0, -1.0)
        return noisy_image

    def crop_center(self, matrix, target_size):
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
