import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

from noise_layers.defocus_blur import DefocusBlur


def test_defocus_blur(image_path, defocus_blur_layer):
    # Load the image
    img = Image.open(image_path).convert("RGB")

    # Apply transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    img_tensor = transform(img).unsqueeze(0)

    # Apply DefocusBlur noise layer
    noised_and_cover = [img_tensor, None]  # Assuming no additional message in your case
    noised_and_cover = defocus_blur_layer(noised_and_cover)

    # Convert tensors to numpy arrays
    original_image = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
    noised_image = noised_and_cover[0].squeeze(0).permute(1, 2, 0).numpy()

    # Display the images
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(noised_image)
    plt.title("Defocus Blurred Image")

    plt.show()

# Example usage
image_path = "test_data/5.png"
defocus_blur_layer = DefocusBlur()
test_defocus_blur(image_path, defocus_blur_layer)
