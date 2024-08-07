from typing import Sequence, Any
import functools
from operator import mul
import json
import zlib

import torch
import torch.nn.functional as F
import numpy as np


def prod(x: Sequence[float]) -> float:
    """Compute the product of elements in a sequence.

    Args:
        x (Sequence[float]): A sequence of integers.

    Returns:
        int: The product of the elements in the sequence.
    """
    return functools.reduce(mul, x, 1)


def rgb_to_ycbcr(rgb_img: torch.Tensor) -> torch.Tensor:
    """Convert an RGB image to YCbCr color space.

    Args:
        rgb_img (torch.Tensor): Input RGB image of shape (3, H, W).

    Returns:
        torch.Tensor: YCbCr image of shape (3, H, W).
    """
    # Transformation matrix from RGB to YCbCr
    transform_matrix = torch.tensor(
        [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]],
        device=rgb_img.device,
    )

    # Offset for Cb and Cr channels
    offset = torch.tensor([0, 128, 128], device=rgb_img.device).view(3, 1, 1)

    # Apply the transformation
    ycbcr_img = offset + torch.einsum(
        "ij, j... -> i...", transform_matrix, rgb_img.float()
    )

    return ycbcr_img


def ycbcr_to_rgb(ycbcr_img: torch.Tensor) -> torch.Tensor:
    """Convert a YCbCr image to RGB color space.

    Args:
        ycbcr_img (torch.Tensor): Input YCbCr image of shape (3, H, W).

    Returns:
        torch.Tensor: RGB image of shape (3, H, W).
    """
    # Transformation matrix from YCbCr to RGB
    transform_matrix = torch.tensor(
        [[1.0, 0.0, 1.40200], [1.0, -0.344136, -0.714136], [1.0, 1.77200, 0.0]],
        device=ycbcr_img.device,
    )

    # The offsets for the Cb and Cr channels
    offset = torch.tensor([0.0, -128.0, -128.0], device=ycbcr_img.device).view(3, 1, 1)

    # Apply the transformation
    rgb_img = torch.einsum(
        "ij, j... -> i...", transform_matrix, ycbcr_img.float() + offset
    )

    return rgb_img


def chroma_downsampling(
    img_ycbcr: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Downsample the chroma channels (Cb and Cr) of a YCbCr image.

    Args:
        img_ycbcr (torch.Tensor): Input YCbCr image of shape (C, H, W).
        **kwargs: Additional arguments for `torch.nn.functional.interpolate`.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Downsampled Y, Cb, and Cr channels.
    """

    # Luminance channel (Y) remains unchanged
    Y = img_ycbcr[0:1, :, :]
    # Downsample Cb channel
    Cb = F.interpolate(img_ycbcr[None, 1:2, :, :], **kwargs).squeeze(0)
    # Downsample Cr channel
    Cr = F.interpolate(img_ycbcr[None, 2:3, :, :], **kwargs).squeeze(0)
    return Y, Cb, Cr


def chroma_upsampling(ycbcr, **kwargs):
    Y, Cb, Cr = ycbcr
    # Upsample Cb channel
    Cb = F.interpolate(Cb[None, :, :], **kwargs).squeeze(0)
    # Upsample Cr channel
    Cr = F.interpolate(Cr[None, :, :], **kwargs).squeeze(0)
    img_ycbcr = torch.cat((Y, Cb, Cr), dim=0)
    return img_ycbcr


def pad_image(
    img: torch.Tensor, patch_size: tuple[int, int], *args, **kwargs
) -> torch.Tensor:
    """Pad an image tensor to ensure it can be evenly divided into patches of size (P, Q).

    Args:
        img (torch.Tensor): The input image tensor of shape (C, H, W).
        patch_size (tuple[int, int]): The height (P) and width (Q) of each patch.
        *args: Additional arguments for `torch.nn.functional.pad`.
        **kwargs: Additional keyword arguments for `torch.nn.functional.pad`.

    Returns:
        Tensor: The padded image tensor.
    """

    _, H, W = img.shape
    P, Q = patch_size
    pad_height = (P - H % P) % P
    pad_width = (Q - W % Q) % Q
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    padded_img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), *args, **kwargs)
    return padded_img


def unpad_image(padded_img: torch.Tensor, orig_size: tuple[int, int]) -> torch.Tensor:
    """Recover the original image from a padded image by removing the added padding.

    Args:
        padded_img (torch.Tensor): The padded image tensor of shape (C, H', W').
        orig_size (tuple[int, int]): The original size of the image before padding (H_orig, W_orig).

    Returns:
        torch.Tensor: The original image tensor of shape (C, H_orig, W_orig).
    """

    _, H_padded, W_padded = padded_img.shape
    H_orig, W_orig = orig_size
    start_h = (H_padded - H_orig) // 2
    end_h = start_h + H_orig
    start_w = (W_padded - W_orig) // 2
    end_w = start_w + W_orig
    original_img = padded_img[:, start_h:end_h, start_w:end_w]
    return original_img


def to_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert a tensor to the specified dtype, clamping its values to be within
    the representable range of the target dtype.

    Args:
        tensor (torch.Tensor): A PyTorch tensor.
        dtype (torch.dtype): The target dtype to convert the tensor to.

    Returns:
        Tensor: A tensor of the target dtype with its values clamped within the representable range.
    """

    # Determine the range of the target dtype
    dtype_min, dtype_max = None, None
    if dtype in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        info = torch.iinfo(dtype)
        dtype_min, dtype_max = info.min, info.max
    elif dtype in (torch.float16, torch.float32, torch.float64):
        info = torch.finfo(dtype)
        dtype_min, dtype_max = info.min, info.max
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Clamp the tensor values and convert to the target dtype
    clamped_tensor = torch.clamp(tensor, dtype_min, dtype_max).to(dtype)

    return clamped_tensor


def quantize(
    tensor: torch.Tensor, target_dtype: torch.dtype
) -> tuple[torch.Tensor, float, float]:
    """Quantize a float tensor to a specified integer data type.

    Args:
        tensor (torch.Tensor): Input tensor of type torch.float32.
        target_dtype (torch.dtype): Target integer data type (e.g., torch.uint8, torch.int16).

    Returns:
        tuple[torch.Tensor, float, float]: Quantized tensor, scale factor, and minimum value of the input tensor.
    """

    # Ensure the tensor is in float32
    assert torch.is_floating_point(tensor), "Tensor must be of type float"

    # Get the range of the target data type
    if target_dtype.is_floating_point:
        info = torch.finfo(target_dtype)
    else:
        info = torch.iinfo(target_dtype)

    qmin, qmax = info.min, info.max

    # Find min and max values in the tensor
    min_val, max_val = tensor.min(), tensor.max()

    # Compute the scale factor
    scale = (max_val - min_val) / (qmax - qmin)

    # Quantize
    quantized = torch.clamp((tensor - min_val) / scale + qmin, qmin, qmax).to(
        target_dtype
    )

    return quantized, scale.item(), min_val.item()


def dequantize(
    quantized_tensor: torch.Tensor, scale: float, min_val: float
) -> torch.Tensor:
    """Dequantize a tensor to float32 values.

    Args:
        quantized_tensor (torch.Tensor): Input quantized tensor.
        scale (float): Scale factor used for quantization.
        min_val (float): Minimum value of the original tensor before quantization.

    Returns:
        Tensor: Dequantized tensor.
    """

    # Convert to float32 for calculation
    quantized_tensor = quantized_tensor.to(torch.float32)

    # Dequantize
    dequantized = (quantized_tensor - quantized_tensor.min()) * scale + min_val

    return dequantized


def _combine_bytes(payload1: bytes, payload2: bytes) -> bytes:
    """Encode two bytes objects into a single bytes object.

    Args:
        payload1 (bytes): First bytes object to encode.
        payload2 (bytes): Second bytes object to encode.

    Returns:
        bytes: A single combined bytes object containing both payloads.
    """

    if not isinstance(payload1, bytes) or not isinstance(payload2, bytes):
        raise TypeError("Both payload1 and payload2 must be bytes objects.")

    # Ensuring the length of payload1 can be represented in 4 bytes (up to 4GB for payload1)
    if len(payload1) > 0xFFFFFFFF:
        raise ValueError("payload1 is too large to encode.")

    len_payload1 = len(payload1).to_bytes(4, byteorder="big")
    return len_payload1 + payload1 + payload2


def _separate_bytes(combined: bytes) -> tuple[bytes, bytes]:
    """Decode the combined bytes object back into two original bytes objects.

    Args:
        combined (bytes): The combined bytes object containing both payloads.

    Returns:
        tuple[bytes, bytes]: The two original bytes objects (payload1, payload2).
    """

    if not isinstance(combined, bytes):
        raise TypeError("Combined must be a bytes object.")

    if len(combined) < 4:
        raise ValueError("Combined data is too short to decode.")

    len_payload1 = int.from_bytes(combined[:4], byteorder="big")
    payload1 = combined[4 : 4 + len_payload1]
    payload2 = combined[4 + len_payload1 :]
    return payload1, payload2


def combine_bytes(payloads: Sequence[bytes]) -> bytes:
    """Combine multiple bytes objects into a single bytes object.

    Args:
        payloads (Sequence[bytes]): A sequence of bytes objects.

    Returns:
        bytes: A single combined bytes object containing all payloads.
    """

    return functools.reduce(_combine_bytes, payloads)


def separate_bytes(combined: bytes, num_payloads: int = 2) -> tuple[bytes, ...]:
    """Split the combined bytes object into its parts (payloads).

    Args:
        combined (bytes): The combined bytes object containing multiple payloads.
        num_payloads (int, optional): The number of payloads. Defaults to 2.

    Returns:
        tuple[bytes, ...]: A tuple containing the original bytes objects (payloads).
    """

    payloads = []
    payload1 = combined
    for _ in range(num_payloads - 1):
        payload1, payload2 = _separate_bytes(payload1)
        payloads.insert(0, payload2)

    payloads.insert(0, payload1)
    return tuple(payloads)


def dict_to_bytes(dictionary: dict) -> bytes:
    """Encode a dictionary into bytes.

    Args:
        dictionary (dict): The dictionary to encode.

    Returns:
        bytes: The encoded dictionary as bytes.
    """

    json_string = json.dumps(dictionary)
    encoded_bytes = json_string.encode("utf-8")
    return encoded_bytes


def bytes_to_dict(encoded_bytes: bytes) -> dict:
    """Decode bytes back into a dictionary.

    Args:
        encoded_bytes (bytes): The encoded dictionary as bytes.

    Returns:
        dict: The decoded dictionary.
    """

    json_string = encoded_bytes.decode("utf-8")
    dictionary = json.loads(json_string)
    return dictionary


def encode_matrix(matrix: torch.Tensor, mode: str = "col") -> bytes:
    """Encode a 2D tensor (matrix) into a compressed bytes object.

    Args:
        matrix (torch.Tensor): A 2D tensor to encode.
        mode (str, optional): Mode of encoding ('col' for column-wise, 'row' for row-wise). Defaults to 'col'.

    Returns:
        bytes: The encoded matrix as a bytes object.
    """

    assert len(matrix.shape) == 2, "'matrix' must be a 2D tensor."
    assert mode in {"col", "row"}, "'mode' must be either 'col' or 'row'."

    if mode == "col":
        fibers = matrix.split(split_size=1, dim=1)

    else:  # row
        fibers = matrix.split(split_size=1, dim=0)

    encoded_fibers = []
    for fiber in fibers:
        encoded_fiber = fiber.numpy().tobytes()
        encoded_fiber = zlib.compress(encoded_fiber, level=9)
        encoded_fibers.append(encoded_fiber)

    metadata = {
        "num_fibers": len(fibers),
        "mode": mode,
        "dtype": str(matrix.dtype).split(".")[-1],
    }
    encoded_metadata = dict_to_bytes(metadata)

    encoded_fibers = combine_bytes(encoded_fibers)
    encoded_matrix = combine_bytes([encoded_metadata, encoded_fibers])

    return encoded_matrix


def decode_matrix(encoded_matrix: bytes, mode: str = "col") -> torch.Tensor:
    """Decode a compressed bytes object back into a 2D tensor (matrix).

    Args:
        encoded_matrix (bytes): The encoded matrix as a bytes object.
        mode (str, optional): Mode of decoding ('col' for column-wise, 'row' for row-wise). Defaults to 'col'.

    Returns:
        torch.Tensor: The decoded 2D tensor.
    """

    assert mode in {"col", "row"}, "'mode' must be either 'col' or 'row'."

    encoded_metadata, encoded_fibers = separate_bytes(encoded_matrix)

    metadata = bytes_to_dict(encoded_metadata)
    num_fibers = metadata["num_fibers"]
    mode = metadata["mode"]
    dtype = metadata["dtype"]

    encoded_fibers = separate_bytes(encoded_fibers, num_payloads=num_fibers)
    fibers = []
    for encoded_fiber in encoded_fibers:
        encoded_fiber = zlib.decompress(encoded_fiber)
        fiber = np.frombuffer(encoded_fiber, dtype=np.dtype(dtype))
        fibers.append(fiber)

    if mode == "col":
        matrix = np.stack(fibers, axis=1)

    else:  # row
        matrix = np.stack(fibers, axis=0)

    return torch.from_numpy(matrix)


def encode_tensor(tensor: torch.Tensor, *args, **kwargs) -> bytes:
    """Encode a PyTorch tensor and its metadata using zlib compression.

    Args:
        tensor (torch.Tensor): The tensor to encode.
        *args: Additional positional arguments for `encode_matrix`.
        **kwargs: Additional keyword arguments for `encode_matrix`.

    Returns:
        bytes: A single bytes object containing both encoded data and metadata.
    """
    if tensor.ndim == 2:
        return encode_matrix(tensor, *args, **kwargs)

    # Convert the tensor to bytes
    encoded_array = tensor.numpy().tobytes()

    # Encode the bytes using a lossless compression
    encoded_array = zlib.compress(encoded_array, level=9)

    # Prepare metadata
    metadata = {"shape": tensor.shape, "dtype": str(tensor.dtype).split(".")[-1]}
    encoded_metadata = dict_to_bytes(metadata)

    # Combine metadata and tensor data into a single bytes object
    encoded_tensor = combine_bytes([encoded_metadata, encoded_array])
    return encoded_tensor


def decode_tensor(encoded_tensor: bytes, *args, **kwargs) -> torch.Tensor:
    """Decode a combined bytes object back into a PyTorch tensor using the encoded shape and dtype.

    Args:
        encoded_tensor (bytes): The combined bytes object containing both encoded data and metadata.
        *args: Additional positional arguments for `decode_matrix`.
        **kwargs: Additional keyword arguments for `decode_matrix`.

    Returns:
        torch.Tensor: The decoded tensor.
    """

    # Extract metadata and array data
    encoded_metadata, encoded_array = separate_bytes(encoded_tensor)

    # Decode metadata
    metadata = bytes_to_dict(encoded_metadata)

    if "num_fibers" in metadata:
        return decode_matrix(encoded_tensor)

    shape = metadata["shape"]
    dtype = metadata["dtype"]

    # Decode the array data
    array = zlib.decompress(encoded_array)

    # Convert back to tensor
    decoded_tensor = torch.from_numpy(
        np.frombuffer(array, dtype=np.dtype(dtype)).reshape(shape)
    )

    return decoded_tensor
