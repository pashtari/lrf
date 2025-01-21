import io

import torch
from torchvision.transforms import functional as FT
from PIL import Image


def pil_encode(image: torch.Tensor, **kwargs) -> bytes:
    """Encode a PyTorch tensor image into bytes using PIL.

    Args:
        image (torch.Tensor): A PyTorch tensor representing the image.
        **kwargs (Any): Additional arguments passed to PIL's save method.

    Returns:
        bytes: The encoded image as bytes.
    """
    # Convert the PyTorch tensor to a PIL image
    pil_image = FT.to_pil_image(image)
    # Encode the PIL image
    buffer = io.BytesIO()
    pil_image.save(buffer, **kwargs)
    return buffer.getvalue()


def pil_decode(encoded_image: bytes) -> torch.Tensor:
    """Decode bytes into a PyTorch tensor image using PIL.

    Args:
        encoded_image (bytes): The encoded image as bytes.

    Returns:
        torch.Tensor: The decoded image as a PyTorch tensor.
    """
    # Decode the bytes to a PIL image
    pil_image = Image.open(io.BytesIO(encoded_image))
    # Convert the PIL image to a PyTorch tensor
    image = FT.pil_to_tensor(pil_image)
    return image
