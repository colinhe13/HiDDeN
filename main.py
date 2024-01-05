import os
import pprint
import argparse
import torch
import pickle
import utils
import logging
import sys

from noise_layers.crop import Crop
from noise_layers.defocus_blur import DefocusBlur
from noise_layers.dropout import Dropout
from noise_layers.motion_blur import MotionBlur
from options import *
from model.hidden import Hidden
from noise_layers.noiser import Noiser
from noise_argparser import NoiseArgParser

from train import train


def main():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 创建一个命令行解析器对象parent_parser
    parent_parser = argparse.ArgumentParser(description='Training of HiDDeN nets')
    subparsers = parent_parser.add_subparsers(dest='command', help='Sub-parser for commands')
    new_run_parser = subparsers.add_parser('new', help='starts a new run')

    # 储存数据的目录
    new_run_parser.add_argument('--data-dir', '-d', required=True, type=str,
                                help='The directory where the data is stored.')
    # 批处理大小
    new_run_parser.add_argument('--batch-size', '-b', required=True, type=int, help='The batch size.')
    # 训练的轮数
    new_run_parser.add_argument('--epochs', '-e', default=300, type=int, help='Number of epochs to run the simulation.')
    # 实验的名称
    new_run_parser.add_argument('--name', required=True, type=str, help='The name of the experiment.')
    # 图片的大小
    new_run_parser.add_argument('--size', '-s', default=128, type=int,
                                help='The size of the images (images are square so this is height and width).')
    # 水印的长度
    new_run_parser.add_argument('--message', '-m', default=30, type=int, help='The length in bits of the watermark.')
    # 从上次实验的文件夹继续
    new_run_parser.add_argument('--continue-from-folder', '-c', default='', type=str,
                                help='The folder from where to continue a previous run. Leave blank if you are starting a new experiment.')
    # parser.add_argument('--tensorboard', dest='tensorboard', action='store_true',
    #                     help='If specified, use adds a Tensorboard log. On by default')
    # 开启日志记录
    new_run_parser.add_argument('--tensorboard', action='store_true',
                                help='Use to switch on Tensorboard logging.')
    # 开启混合精度训练
    new_run_parser.add_argument('--enable-fp16', dest='enable_fp16', action='store_true',
                                help='Enable mixed-precision training.')
    # 噪声层的配置
    new_run_parser.add_argument('--noise', nargs='*', action=NoiseArgParser,
                                help="Noise layers configuration. Use quotes when specifying configuration, e.g. 'cropout((0.55, 0.6), (0.55, 0.6))'")

    new_run_parser.set_defaults(tensorboard=False)
    new_run_parser.set_defaults(enable_fp16=False)

    # 继续训练的相关配置
    continue_parser = subparsers.add_parser('continue', help='Continue a previous run')
    continue_parser.add_argument('--folder', '-f', required=True, type=str,
                                 help='Continue from the last checkpoint in this folder.')
    continue_parser.add_argument('--data-dir', '-d', required=False, type=str,
                                 help='The directory where the data is stored. Specify a value only if you want to override the previous value.')
    continue_parser.add_argument('--epochs', '-e', required=False, type=int,
                                help='Number of epochs to run the simulation. Specify a value only if you want to override the previous value.')
    # continue_parser.add_argument('--tensorboard', action='store_true',
    #                             help='Override the previous setting regarding tensorboard logging.')

    args = parent_parser.parse_args()
    checkpoint = None
    loaded_checkpoint_file_name = None

    # 如果是继续训练
    if args.command == 'continue':
        this_run_folder = args.folder
        options_file = os.path.join(this_run_folder, 'options-and-config.pickle')
        train_options, hidden_config, noise_config = utils.load_options(options_file)
        checkpoint, loaded_checkpoint_file_name = utils.load_last_checkpoint(os.path.join(this_run_folder, 'checkpoints'))
        train_options.start_epoch = checkpoint['epoch'] + 1
        if args.data_dir is not None:
            train_options.train_folder = os.path.join(args.data_dir, 'train')
            train_options.validation_folder = os.path.join(args.data_dir, 'val')
        if args.epochs is not None:
            if train_options.start_epoch < args.epochs:
                train_options.number_of_epochs = args.epochs
            else:
                print(f'Command-line specifies of number of epochs = {args.epochs}, but folder={args.folder} '
                      f'already contains checkpoint for epoch = {train_options.start_epoch}.')
                exit(1)

    # 如果是新的实验
    else:
        assert args.command == 'new'
        # 设置起始轮数为1
        start_epoch = 1
        # 创建一个新的训练选项对象
        train_options = TrainingOptions(
            batch_size=args.batch_size,
            number_of_epochs=args.epochs,
            train_folder=os.path.join(args.data_dir, 'train'),
            validation_folder=os.path.join(args.data_dir, 'val'),
            runs_folder=os.path.join('.', 'runs'),
            start_epoch=start_epoch,
            experiment_name=args.name)
        # 创建一个新的噪声层配置对象
        noise_config = args.noise if args.noise is not None else []
        # 创建一个新的隐藏层配置对象
        hidden_config = HiDDenConfiguration(H=args.size, W=args.size,
                                            message_length=args.message,
                                            encoder_blocks=4, encoder_channels=64,
                                            decoder_blocks=7, decoder_channels=64,
                                            use_discriminator=True,
                                            use_vgg=False,
                                            discriminator_blocks=3, discriminator_channels=64,
                                            decoder_loss=1,
                                            encoder_loss=0.7,
                                            adversarial_loss=1e-3,
                                            enable_fp16=args.enable_fp16
                                            )
        # 创建一个新的实验文件夹
        this_run_folder = utils.create_folder_for_run(train_options.runs_folder, args.name)
        # 将训练选项、噪声层配置、隐藏层配置保存到文件中
        with open(os.path.join(this_run_folder, 'options-and-config.pickle'), 'wb+') as f:
            pickle.dump(train_options, f)
            pickle.dump(noise_config, f)
            pickle.dump(hidden_config, f)

    # 设置日志记录的格式
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            # 将日志记录到文件中
                            logging.FileHandler(os.path.join(this_run_folder, f'{train_options.experiment_name}.log')),
                            # 将日志记录到控制台中
                            logging.StreamHandler(sys.stdout)
                        ])
    # 如果开启了tensorboard，则创建tensorboard日志记录器
    if (args.command == 'new' and args.tensorboard) or \
            (args.command == 'continue' and os.path.isdir(os.path.join(this_run_folder, 'tb-logs'))):
        logging.info('Tensorboard is enabled. Creating logger.')
        from tensorboard_logger import TensorBoardLogger
        tb_logger = TensorBoardLogger(os.path.join(this_run_folder, 'tb-logs'))
    else:
        tb_logger = None

    # 创建一个新的噪声层对象
    noiser = Noiser(noise_config, device)
    # 创建一个Hidden对象，用于处理HiDDeN模型的训练和推断，同时传入隐藏层配置、计算设备、噪声生成器和TensorBoard记录器
    model = Hidden(hidden_config, device, noiser, tb_logger)

    # 如果是继续训练，则加载模型参数
    if args.command == 'continue':
        # if we are continuing, we have to load the model params
        assert checkpoint is not None
        logging.info(f'Loading checkpoint from file {loaded_checkpoint_file_name}')
        utils.model_from_checkpoint(model, checkpoint)

    # 打印HiDDeN模型的字符串表示形式
    logging.info('HiDDeN model: {}\n'.format(model.to_stirng()))
    # 打印模型配置的信息
    logging.info('Model Configuration:\n')
    logging.info(pprint.pformat(vars(hidden_config)))
    # 打印噪声层配置的信息
    logging.info('\nNoise configuration:\n')
    logging.info(pprint.pformat(str(noise_config)))
    # 打印训练选项的信息
    logging.info('\nTraining train_options:\n')
    logging.info(pprint.pformat(vars(train_options)))

    # 进行训练
    train(model, device, hidden_config, train_options, this_run_folder, tb_logger)


if __name__ == '__main__':
    main()


    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # hidden_config = HiDDenConfiguration(H=16, W=16,
    #                                     # message_length=52428,
    #                                     message_length=20,
    #                                     encoder_blocks=4, encoder_channels=64,
    #                                     decoder_blocks=7, decoder_channels=64,
    #                                     use_discriminator=True,
    #                                     use_vgg=False,
    #                                     discriminator_blocks=3, discriminator_channels=64,
    #                                     decoder_loss=1,
    #                                     encoder_loss=0.7,
    #                                     adversarial_loss=1e-3,
    #                                     enable_fp16=False
    #                                     )
    # noiser = Noiser([DefocusBlur(), MotionBlur()], device)
    # # noiser = Noiser([Crop([0.11,0.22],[0.33,0.44])], device)
    # tb_logger = None
    # model = Hidden(hidden_config, device, noiser, tb_logger)
    # train_options = TrainingOptions(
    #     # batch_size=12,
    #     batch_size=2,
    #     number_of_epochs=100,
    #     # train_folder=os.path.join('data', 'boss_h', 'data_size512', '500_50', 'train'),
    #     # validation_folder=os.path.join('data', 'boss_h', 'data_size512', '500_50', 'val'),
    #     # train_folder=os.path.join('data', 'boss_h', 'data_size512_pgm', '5_1', 'train'),
    #     # validation_folder=os.path.join('data', 'boss_h', 'data_size512_pgm', '5_1', 'val'),
    #     train_folder=os.path.join('data', 'boss', 'data_size512_pgm', '5_2', 'train'),
    #     validation_folder=os.path.join('data', 'boss', 'data_size512_pgm', '5_2', 'val'),
    #     runs_folder=os.path.join('.', 'runs'),
    #     start_epoch=1,
    #     experiment_name='halftone-noise-size16-5'
    # )
    # this_run_folder = utils.create_folder_for_run(train_options.runs_folder, train_options.experiment_name)
    # with open(os.path.join(this_run_folder, 'options-and-config.pickle'), 'wb+') as f:
    #     pickle.dump(train_options, f)
    #     pickle.dump(hidden_config, f)
    # train(model, device, hidden_config, train_options, this_run_folder, tb_logger)


"""
python main.py new --name no-noise-size512-500 --data-dir data/boss_h/data_size512/500_50 --epochs 100 --size 512 --message 52428 --batch-size 12 
python main.py new --name no-noise-size16-5000 --data-dir data/boss_h/data_size512_pgm/5000_500 --epochs 200 --size 16 --message 50 --batch-size 12
python main.py new --name no-noise-size128-5000 --data-dir data/boss_h/data_size512_pgm/5000_500 --epochs 300 --size 128 --message 30 --batch-size 12 --noise 'motion+defocus'; shutdown
python main.py new --name motion-defocus-size128-5000 --data-dir data/boss_h/data_size512_pgm/5000_500 --epochs 300 --size 128 --message 30 --batch-size 12 --noise 'motion+defocus'; shutdown

"""