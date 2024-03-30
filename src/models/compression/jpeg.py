from torchvision.transforms import functional as F
from PIL import Image
import io


def jpeg_encode(image_tensor, **kwargs):
    # Convert the PyTorch tensor to a PIL image
    pil_image = F.to_pil_image(image_tensor)
    # Encode the PIL image to JPEG bytes
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format="JPEG", **kwargs)
    return img_byte_arr.getvalue()


def jpeg_decode(jpeg_bytes):
    # Decode the JPEG bytes to a PIL image
    pil_image = Image.open(io.BytesIO(jpeg_bytes))
    # Convert the PIL image to a PyTorch tensor
    image_tensor = F.pil_to_tensor(pil_image)
    return image_tensor


# import torch
# import matplotlib.pyplot as plt
# from skimage import data

# # Load the astronaut image from skimage and display it
# image = data.astronaut()
# image = torch.from_numpy(image).permute(2, 0, 1)

# # Encode and then decode the image
# encoded_image = jpeg_encode(image, quality=10)
# decoded_image = jpeg_decode(encoded_image)

# # Visualize the original and reconstructed images
# fig, ax = plt.subplots(1, 2, figsize=(12, 5))
# ax[0].imshow(image.permute(1, 2, 0))
# ax[0].set_title("Original Image")
# ax[0].axis("off")

# ax[1].imshow(decoded_image.permute(1, 2, 0))
# ax[1].set_title("Compressed Image")
# ax[1].axis("off")

# plt.tight_layout()
# plt.show()
