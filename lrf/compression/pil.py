import io

import torch
from torchvision.transforms import functional as FT
from PIL import Image
from lrf.compression.utils import combine_bytes, separate_bytes


def pil_encode(image, **kwargs):
    # Convert the PyTorch tensor to a PIL image
    pil_image = FT.to_pil_image(image)
    # Encode the PIL image
    buffer = io.BytesIO()
    pil_image.save(buffer, **kwargs)
    return buffer.getvalue()


def pil_decode(encoded_image):
    # Decode the bytes to a PIL image
    pil_image = Image.open(io.BytesIO(encoded_image))
    # Convert the PIL image to a PyTorch tensor
    image = FT.pil_to_tensor(pil_image)
    return image


def batched_pil_encode(images, **kwargs):
    return combine_bytes([pil_encode(image, **kwargs) for image in images])


def batched_pil_decode(encoded_images, num_images):
    encoded_images = separate_bytes(encoded_images, num_payloads=num_images)
    return torch.stack([pil_decode(enc_img) for enc_img in encoded_images])
