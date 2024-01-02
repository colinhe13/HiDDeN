from PIL import Image
import torch
from torchvision import transforms



def load_pgm_to_tensor(file_path):
    # 使用 Pillow 打开 PGM 文件
    image = Image.open(file_path)

    # 应用定义的转换
    tensor_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])(image)

    return tensor_image

# 示例用法：
pgm_file_path = "boss_h/data_size512_pgm/1_14_17_5.pgm"
tensor_image = load_pgm_to_tensor(pgm_file_path)
print(tensor_image.shape)
print(tensor_image)