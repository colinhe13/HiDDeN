import torch
import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu


class Encoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    """
    相关参数
    config.encoder_blocks = 3
    config.encoder_channels = conv_channels = 64
    """
    def __init__(self, config: HiDDenConfiguration):
        super(Encoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks

        # 创建卷积层列表，首先添加一个输入通道数为3，输出通道数为64的卷积层
        layers = [ConvBNRelu(3, self.conv_channels)]

        # 添加卷积层，每个卷积层的输入通道数和输出通道数都是64
        for _ in range(config.encoder_blocks-1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        # 将所有的卷积层组合成一个序列
        self.conv_layers = nn.Sequential(*layers)
        # 添加一个卷积层，用于拼接输入的图片和水印
        self.after_concat_layer = ConvBNRelu(self.conv_channels + 3 + config.message_length,
                                             self.conv_channels)

        # 添加一个卷积层，输入通道数为64，输出通道数为3，用于将这些特征转换为一个RGB图片
        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)

    def forward(self, image, message):

        # First, add two dummy dimensions in the end of the message.
        # This is required for the .expand to work correctly
        # 在消息的末尾添加两个虚拟维度，这是为了使.expand能够正确工作
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)

        # 使用 .expand 方法将消息的维度扩展为与图像相同
        expanded_message = expanded_message.expand(-1,-1, self.H, self.W)
        # 将图片输入到卷积层中
        encoded_image = self.conv_layers(image)
        # concatenate expanded message and image
        # 拼接扩展的消息、编码图像和原始图像，生成一个包含扩展消息 编码后的图像 原始图像的张量，其形状为 (batch_size, message_length + conv_channels + 3, H, W)
        concat = torch.cat([expanded_message, encoded_image, image], dim=1)
        # 将拼接后的结果输入到卷积层中
        im_w = self.after_concat_layer(concat)
        # 通过最终卷积层输出一个RGB图片
        im_w = self.final_layer(im_w)
        return im_w
