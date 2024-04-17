import io

from torchvision.transforms import functional as FT
from PIL import Image


def pil_encode(image_tensor, **kwargs):
    # Convert the PyTorch tensor to a PIL image
    pil_image = FT.to_pil_image(image_tensor)
    # Encode the PIL image to JPEG bytes
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, **kwargs)
    return img_byte_arr.getvalue()


def pil_decode(bytes):
    # Decode the bytes to a PIL image
    pil_image = Image.open(io.BytesIO(bytes))
    # Convert the PIL image to a PyTorch tensor
    image_tensor = FT.pil_to_tensor(pil_image)
    return image_tensor
