import numpy as np
import torch
import torch.nn as nn

from options import HiDDenConfiguration
from model.discriminator import Discriminator
from model.encoder_decoder import EncoderDecoder
from vgg_loss import VGGLoss
from noise_layers.noiser import Noiser


class Hidden:
    def __init__(self, configuration: HiDDenConfiguration, device: torch.device, noiser: Noiser, tb_logger):
        """
        :param configuration: Configuration for the net, such as the size of the input image, number of channels in the intermediate layers, etc.
        :param device: torch.device object, CPU or GPU
        :param noiser: Object representing stacked noise layers.
        :param tb_logger: Optional TensorboardX logger object, if specified -- enables Tensorboard logging
        """
        super(Hidden, self).__init__()

        self.encoder_decoder = EncoderDecoder(configuration, noiser).to(device)
        self.discriminator = Discriminator(configuration).to(device)
        self.optimizer_enc_dec = torch.optim.Adam(self.encoder_decoder.parameters())
        self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters())

        if configuration.use_vgg:
            self.vgg_loss = VGGLoss(3, 1, False)
            self.vgg_loss.to(device)
        else:
            self.vgg_loss = None

        self.config = configuration
        self.device = device

        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)

        # Defined the labels used for training the discriminator/adversarial loss
        self.cover_label = 1
        self.encoded_label = 0

        self.tb_logger = tb_logger
        if tb_logger is not None:
            from tensorboard_logger import TensorBoardLogger
            encoder_final = self.encoder_decoder.encoder._modules['final_layer']
            encoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/encoder_out'))
            decoder_final = self.encoder_decoder.decoder._modules['linear']
            decoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/decoder_out'))
            discrim_final = self.discriminator._modules['linear']
            discrim_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/discrim_out'))


    def train_on_batch(self, batch: list):
        """
        Trains the network on a single batch consisting of images and messages
        :param batch: batch of training data, in the form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        """
        为什么先训练判别器，再训练生成器？这是GAN网络，生成器和判别器是对抗关系，一次循环中先生成哪个没有区别
        VGG损失是什么？
        """
        # 从批次中解包得到图像和消息
        images, messages = batch

        # 获取批次大小
        batch_size = images.shape[0]
        # 将编码器-解码器和判别器设置为训练模式
        self.encoder_decoder.train()
        self.discriminator.train()
        # 开启梯度计算
        with torch.enable_grad():
            # ---------------- Train the discriminator -----------------------------
            # 将判别器的梯度置零
            self.optimizer_discrim.zero_grad()

            # train on cover
            # ground truth 创建用于判别器训练的目标标签，表示真实图像（一个batch_size*1的张量，每个元素的值都是1）
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            # 创建用于判别器训练的目标标签，表示编码后的图像（一个batch_size*1的张量，每个元素的值都是0）
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            # 创建用于生成器（编码器-解码器）训练的目标标签，表示编码后的图像（一个batch_size*1的张量，每个元素的值都是1）
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)

            # 判别器对真实图像进行前向传播
            d_on_cover = self.discriminator(images)
            # 计算判别器对真实图像的二进制交叉熵损失
            d_target_label_cover = d_target_label_cover.float()
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)
            # 反向传播并计算判别器对真实图像的梯度
            d_loss_on_cover.backward()

            # train on fake
            # 编码器-解码器对图像和消息进行前向传播，生成编码后的图像、加噪图像和解码后的消息
            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
            # 判别器对编码后的图像进行前向传播，使用.detach()防止梯度传播到编码器-解码器
            d_on_encoded = self.discriminator(encoded_images.detach())
            # 计算判别器对编码后的图像的二进制交叉熵损失
            d_target_label_encoded = d_target_label_encoded.float()
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)

            # 反向传播并计算判别器对编码后的图像的梯度
            d_loss_on_encoded.backward()
            # 更新判别器的参数
            self.optimizer_discrim.step()

            # --------------Train the generator (encoder-decoder) ---------------------
            # 清零编码器-解码器的梯度
            self.optimizer_enc_dec.zero_grad()
            # target label for encoded images should be 'cover', because we want to fool the discriminator
            # 判别器对编码后的图像进行前向传播，用于生成器的训练
            d_on_encoded_for_enc = self.discriminator(encoded_images)
            # 计算生成器对判别器的对抗损失
            g_target_label_encoded = g_target_label_encoded.float()
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)

            # 如果没有VGG损失
            if self.vgg_loss == None:
                # 计算生成器对编码图像的均方误差损失
                g_loss_enc = self.mse_loss(encoded_images, images)
            # 如果有VGG损失
            else:
                # 使用VGG损失计算真实图像的特征
                vgg_on_cov = self.vgg_loss(images)
                # 使用VGG损失计算编码后的图像的特征
                vgg_on_enc = self.vgg_loss(encoded_images)
                # 计算生成器对编码图像的均方误差损失
                g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)

            # 计算生成器对解码消息的均方误差损失
            g_loss_dec = self.mse_loss(decoded_messages, messages)
            # 计算生成器的总损失
            g_loss = self.config.adversarial_loss * g_loss_adv + self.config.encoder_loss * g_loss_enc \
                     + self.config.decoder_loss * g_loss_dec

            # 反向传播并计算生成器的梯度
            g_loss.backward()
            # 更新生成器（编码器-解码器）的参数
            self.optimizer_enc_dec.step()

        # 将解码后的消息进行四舍五入，得到二进制消息
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        # 计算二进制消息的平均误差
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        # 构建各种损失指标的字典
        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item()
        }
        # 返回损失指标字典和生成器生成的图像、加噪图像和解码后的消息
        return losses, (encoded_images, noised_images, decoded_messages)

    def validate_on_batch(self, batch: list):
        """
        Runs validation on a single batch of data consisting of images and messages
        :param batch: batch of validation data, in form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        # if TensorboardX logging is enabled, save some of the tensors.
        # 如果启用了TensorboardX日志记录，将一些张量保存到TensorboardX日志文件中
        if self.tb_logger is not None:
            encoder_final = self.encoder_decoder.encoder._modules['final_layer']
            self.tb_logger.add_tensor('weights/encoder_out', encoder_final.weight)
            decoder_final = self.encoder_decoder.decoder._modules['linear']
            self.tb_logger.add_tensor('weights/decoder_out', decoder_final.weight)
            discrim_final = self.discriminator._modules['linear']
            self.tb_logger.add_tensor('weights/discrim_out', discrim_final.weight)

        # 从验证批次中解包得到图像和消息
        images, messages = batch

        # 获取批次大小
        batch_size = images.shape[0]

        # 将编码器-解码器和判别器设置为验证模式
        self.encoder_decoder.eval()
        self.discriminator.eval()
        # 关闭梯度计算
        with torch.no_grad():
            # 创建用于判别器训练的目标标签，表示真实图像
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            # 创建用于判别器训练的目标标签，表示编码后的图像
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            # 创建用于生成器（编码器-解码器）训练的目标标签，表示编码后的图像
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)

            # 判别器对真实图像进行前向传播
            d_on_cover = self.discriminator(images)
            # 计算判别器对真实图像的二进制交叉熵损失
            d_target_label_cover = d_target_label_cover.float()
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)

            # 编码器-解码器对图像和消息进行前向传播，生成编码后的图像、加噪图像和解码后的消息
            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

            # 判别器对编码后的图像进行前向传播，使用.detach()防止梯度传播到编码器-解码器
            d_on_encoded = self.discriminator(encoded_images)
            # 计算判别器对编码后的图像的二进制交叉熵损失
            d_target_label_encoded = d_target_label_encoded.float()
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)

            # 判别器对编码后的图像进行前向传播，用于生成器的训练
            d_on_encoded_for_enc = self.discriminator(encoded_images)
            # 计算生成器对判别器的对抗损失
            g_target_label_encoded = g_target_label_encoded.float()
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)

            # 如果没有VGG损失
            if self.vgg_loss is None:
                # 计算生成器对编码图像的均方误差损失
                g_loss_enc = self.mse_loss(encoded_images, images)
            # 如果有VGG损失
            else:
                # 使用VGG损失计算真实图像的特征
                vgg_on_cov = self.vgg_loss(images)
                # 使用VGG损失计算编码后的图像的特征
                vgg_on_enc = self.vgg_loss(encoded_images)
                # 计算生成器对编码图像的均方误差损失
                g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)

            # 计算生成器对解码消息的均方误差损失
            g_loss_dec = self.mse_loss(decoded_messages, messages)
            # 计算生成器的总损失
            g_loss = self.config.adversarial_loss * g_loss_adv + self.config.encoder_loss * g_loss_enc \
                     + self.config.decoder_loss * g_loss_dec

        # 将解码后的消息进行四舍五入，得到二进制消息
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        # 计算二进制消息的平均误差
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        # 构建各种损失指标的字典
        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item()
        }
        # 返回损失指标字典和生成器生成的图像、加噪图像和解码后的消息
        return losses, (encoded_images, noised_images, decoded_messages)

    def to_stirng(self):
        return '{}\n{}'.format(str(self.encoder_decoder), str(self.discriminator))
