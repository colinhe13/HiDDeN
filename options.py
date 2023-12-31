# 定义了一些配置类，用于配置训练过程中的参数
class TrainingOptions:
    """
    Configuration options for the training
    """

    def __init__(self,
                 batch_size: int,
                 number_of_epochs: int,
                 train_folder: str, validation_folder: str, runs_folder: str,
                 start_epoch: int, experiment_name: str):
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.train_folder = train_folder
        self.validation_folder = validation_folder
        self.runs_folder = runs_folder
        self.start_epoch = start_epoch
        self.experiment_name = experiment_name

    def __str__(self):
        return f'batch_size={self.batch_size}, number_of_epochs={self.number_of_epochs}, ' \
               f'train_folder={self.train_folder}, validation_folder={self.validation_folder}, ' \
               f'runs_folder={self.runs_folder}, start_epoch={self.start_epoch}'
               # f'runs_folder={self.runs_folder}, start_epoch={self.start_epoch}, experiment_name={self.experiment_name}'

class HiDDenConfiguration():
    """
    The HiDDeN network configuration.
    """

    def __init__(self, H: int, W: int, message_length: int,
                 encoder_blocks: int, encoder_channels: int,
                 decoder_blocks: int, decoder_channels: int,
                 use_discriminator: bool,
                 use_vgg: bool,
                 discriminator_blocks: int, discriminator_channels: int,
                 decoder_loss: float,
                 encoder_loss: float,
                 adversarial_loss: float,
                 enable_fp16: bool = False):
        self.H = H
        self.W = W
        self.message_length = message_length
        self.encoder_blocks = encoder_blocks
        self.encoder_channels = encoder_channels
        self.use_discriminator = use_discriminator
        self.use_vgg = use_vgg
        self.decoder_blocks = decoder_blocks
        self.decoder_channels = decoder_channels
        self.discriminator_blocks = discriminator_blocks
        self.discriminator_channels = discriminator_channels
        self.decoder_loss = decoder_loss
        self.encoder_loss = encoder_loss
        self.adversarial_loss = adversarial_loss
        self.enable_fp16 = enable_fp16

    def __str__(self):
        return f'H={self.H}, W={self.W}, message_length={self.message_length}, ' \
               f'encoder_blocks={self.encoder_blocks}, encoder_channels={self.encoder_channels}, ' \
               f'use_discriminator={self.use_discriminator}, use_vgg={self.use_vgg}, ' \
               f'decoder_blocks={self.decoder_blocks}, decoder_channels={self.decoder_channels}, ' \
               f'discriminator_blocks={self.discriminator_blocks}, discriminator_channels={self.discriminator_channels}, ' \
               f'decoder_loss={self.decoder_loss}, encoder_loss={self.encoder_loss}, ' \
               f'adversarial_loss={self.adversarial_loss}'
               # f'adversarial_loss={self.adversarial_loss}, enable_fp16={self.enable_fp16}'
