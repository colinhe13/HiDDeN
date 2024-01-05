import numpy as np
import torch.nn as nn
from noise_layers.identity import Identity
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.quantization import Quantization
from noise_layers.screen import ThresholdNoise


class Noiser(nn.Module):
    """
    This module allows to combine different noise layers into a sequential noise module. The
    configuration and the sequence of the noise layers is controlled by the noise_config parameter.
    """
    def __init__(self, noise_layers: list, device):
        super(Noiser, self).__init__()
        # 初始化噪声层列表，包含一个Identity空噪声层
        self.noise_layers = [Identity()]
        # 初始化噪声层列表，添加半色调噪声层
        # self.noise_layers = [ThresholdNoise(threshold_matrix_path="data/screen_matrx_512.npy")]
        # 遍历噪声层列表，根据字符串选择噪声层
        for layer in noise_layers:
            # 如果是字符串
            if type(layer) is str:
                # 添加一个JPEG压缩噪声层
                if layer == 'JpegPlaceholder':
                    self.noise_layers.append(JpegCompression(device))
                # 添加一个量化噪声层
                elif layer == 'QuantizationPlaceholder':
                    self.noise_layers.append(Quantization(device))
                # 抛出异常
                else:
                    raise ValueError(f'Wrong layer placeholder string in Noiser.__init__().'
                                     f' Expected "JpegPlaceholder" or "QuantizationPlaceholder" but got {layer} instead')
            # 直接添加噪声层
            else:
                self.noise_layers.append(layer)
        # self.noise_layers = nn.Sequential(*noise_layers)

    def forward(self, encoded_and_cover):
        random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
        return random_noise_layer(encoded_and_cover)

