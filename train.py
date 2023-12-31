import os
import time
import torch
import numpy as np
import utils
import logging
from collections import defaultdict

from options import *
from model.hidden import Hidden
from average_meter import AverageMeter


def train(model: Hidden,
          device: torch.device,
          hidden_config: HiDDenConfiguration,
          train_options: TrainingOptions,
          this_run_folder: str,
          tb_logger):
    """
    Trains the HiDDeN model
    :param model: The model
    :param device: torch.device object, usually this is GPU (if avaliable), otherwise CPU.
    :param hidden_config: The network configuration
    :param train_options: The training settings
    :param this_run_folder: The parent folder for the current training run to store training artifacts/results/logs.
    :param tb_logger: TensorBoardLogger object which is a thin wrapper for TensorboardX logger.
                Pass None to disable TensorboardX logging
    :return:
    """

    # 获取训练数据加载器和验证数据加载器（具体读出的数据是什么？）
    train_data, val_data = utils.get_data_loaders(hidden_config, train_options)
    # 获取训练数据集中的文件数量
    file_count = len(train_data.dataset)
    # 计算每个epoch中的步数
    if file_count % train_options.batch_size == 0:
        steps_in_epoch = file_count // train_options.batch_size
    else:
        steps_in_epoch = file_count // train_options.batch_size + 1

    # 每隔10个步骤打印一次日志
    print_each = 10
    # 保存8张图片
    images_to_save = 8
    # 设置保存图片的大小
    saved_images_size = (512, 512)

    # 开始训练，共循环迭代number_of_epochs - start_epoch次
    for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):
        # 打印开始信息
        logging.info('\nStarting epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        # 打印批次大小和每个epoch中的步数
        logging.info('Batch size = {}\nSteps in epoch = {}'.format(train_options.batch_size, steps_in_epoch))
        # 创建一个默认值为AverageMeter对象的字典，存储训练损失的统计信息
        training_losses = defaultdict(AverageMeter)
        # 记录开始时间
        epoch_start = time.time()
        step = 1
        for image, _ in train_data:
            # 将图片移到计算设备上
            image = image.to(device)
            # 随机生成与图像大小相同的消息，移到计算设备上
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
            # 在当前批次上进行训练，获取损失
            losses, _ = model.train_on_batch([image, message])

            # 更新损失统计信息
            for name, loss in losses.items():
                training_losses[name].update(loss)
            # 如果当前步骤是10的倍数，或者当前步骤是当前epoch中的最后一步，则打印当前epoch的训练损失统计信息，记录训练进度
            if step % print_each == 0 or step == steps_in_epoch:
                logging.info(
                    'Epoch: {}/{} Step: {}/{}'.format(epoch, train_options.number_of_epochs, step, steps_in_epoch))
                utils.log_progress(training_losses)
                logging.info('-' * 40)
            step += 1

        # 记录当前epoch的训练损失统计信息
        train_duration = time.time() - epoch_start
        logging.info('Epoch {} training duration {:.2f} sec'.format(epoch, train_duration))
        logging.info('-' * 40)
        # 将当前epoch的训练损失统计信息写入文件中
        utils.write_losses(os.path.join(this_run_folder, 'train.csv'), training_losses, epoch, train_duration)
        # 如果开启了tensorboard，则将当前epoch的训练损失统计信息写入tensorboard日志文件中
        if tb_logger is not None:
            tb_logger.save_losses(training_losses, epoch)
            tb_logger.save_grads(epoch)
            tb_logger.save_tensors(epoch)

        # 用于控制是否保存生成的图像
        first_iteration = True
        # 创建一个默认值为AverageMeter对象的字典，存储验证损失的统计信息
        validation_losses = defaultdict(AverageMeter)
        # 打印验证信息
        logging.info('Running validation for epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        # 遍历验证数据集中的每个批次
        for image, _ in val_data:
            # 将图片移到计算设备上
            image = image.to(device)
            # 随机生成与图像大小相同的消息，移到计算设备上
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
            # 在当前批次上进行验证，获取损失
            losses, (encoded_images, noised_images, decoded_messages) = model.validate_on_batch([image, message])
            # 更新损失统计信息
            for name, loss in losses.items():
                validation_losses[name].update(loss)
            # 如果是这个epoch中第一次迭代，则保存生成的图像
            if first_iteration:
                if hidden_config.enable_fp16:
                    image = image.float()
                    encoded_images = encoded_images.float()
                utils.save_images(image.cpu()[:images_to_save, :, :, :],
                                  encoded_images[:images_to_save, :, :, :].cpu(),
                                  epoch,
                                  os.path.join(this_run_folder, 'images'), resize_to=saved_images_size)
                first_iteration = False
        # 打印验证损失统计信息，记录验证进度
        utils.log_progress(validation_losses)
        logging.info('-' * 40)
        # 保存检查点文件，包括模型参数、实验名称、当前epoch、检查点文件夹路径
        utils.save_checkpoint(model, train_options.experiment_name, epoch, os.path.join(this_run_folder, 'checkpoints'))
        # 将当前epoch的验证损失统计信息写入csv文件中，记录验证损失，epoch，验证时间
        utils.write_losses(os.path.join(this_run_folder, 'validation.csv'), validation_losses, epoch,
                           time.time() - epoch_start)



        """
            get_data_loaders(hidden_config, train_options)函数的作用是获取训练数据加载器和验证数据加载器，具体读出的数据是什么
            message是什么，有什么用？只是用于训练吗？训练后的模型可以根据需要输入任意信息，实现隐藏？
            每一次迭代中都交替进行训练和验证的目的：可以监控过拟合，调整超参数，提前停止训练
        """