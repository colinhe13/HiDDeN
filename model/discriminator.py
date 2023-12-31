import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu

class Discriminator(nn.Module):
    """
    Discriminator network. Receives an image and has to figure out whether it has a watermark inserted into it, or not.
    """

    """
    输入通道数3和图片的关系是什么？图片是3通道的，所以输入通道数为3
    """

    """
    相关参数
    config.discriminator_blocks=3
    config.discriminator_channels=64
    """

    def __init__(self, config: HiDDenConfiguration):
        # 初始化父类
        super(Discriminator, self).__init__()

        # 定义卷积层
        # 首先添加一个输入通道数为3，输出通道数为64
        layers = [ConvBNRelu(3, config.discriminator_channels)]
        # 然后添加2个卷积层,每个卷积层的输入通道数和输出通道数都是64
        for _ in range(config.discriminator_blocks-1):
            layers.append(ConvBNRelu(config.discriminator_channels, config.discriminator_channels))

        # 添加一个全局平均池化层,输出通道数为1,输入通道数为64,池化核大小为1,输出大小为1*1,用于将卷积层输出的特征图变成全局平均（全局平均池化层不改变通道数，只是在空间维度上取平均）
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        # 将所有的层组合成一个序列
        self.before_linear = nn.Sequential(*layers)
        # 添加一个线性层,输入通道数为64,输出通道数为1,用于将这些特征转换为一个标量输出
        self.linear = nn.Linear(config.discriminator_channels, 1)

    def forward(self, image):
        # 将图片输入到卷积层中
        X = self.before_linear(image)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        X.squeeze_(3).squeeze_(2)
        # 将卷积层的输出输入到线性层中，得到一个标量输出，用于判断图片是否有水印
        X = self.linear(X)
        # X = torch.sigmoid(X)
        return X