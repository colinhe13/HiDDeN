import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from noise_layers.defocus_blur import DefocusBlur
from noise_layers.gaussian_blur import GaussianBlur
from noise_layers.motion_blur import MotionBlur

# 读取测试图像
image_path = 'test_data/5.png'
original_image = Image.open(image_path).convert("RGB")

# 定义图像转换
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 将图像转换为 PyTorch 张量
input_image = transform(original_image).unsqueeze(0)

# 创建模型和噪声层
# your_network = YourNetwork()
defocus_blur = DefocusBlur()
motion_blur = MotionBlur()
gaussian_blur = GaussianBlur(kernel_size=15, sigma=3)


# 应用散焦模糊
blurred_image_defocus = defocus_blur(input_image.clone())

# 应用高斯模糊
blurred_image_gaussian = gaussian_blur(input_image.clone())

# 应用运动模糊
blurred_image_motion = motion_blur(input_image.clone())

# 可视化结果
plt.figure(figsize=(15, 8))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(original_image)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Defocus Blur")
plt.imshow(blurred_image_defocus.squeeze(0).permute(1, 2, 0).numpy())
plt.axis('off')

# plt.subplot(1, 3, 2)
# plt.title("Gaussian Blur")
# plt.title("Defocus Blur")
# plt.imshow(blurred_image_gaussian.squeeze(0).permute(1, 2, 0).numpy())
# plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Motion Blur")
plt.imshow(blurred_image_motion.squeeze(0).permute(1, 2, 0).numpy())
plt.axis('off')

plt.show()
