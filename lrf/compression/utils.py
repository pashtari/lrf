from typing import Sequence
import functools
from operator import mul
import json
import zlib
import io

import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import joblib


def prod(x):
    return functools.reduce(mul, x, 1)


def zscore_normalize(tensor, dim=(-2, -1), eps=1e-8):
    mean = torch.mean(tensor, dim=dim, keepdim=True)
    std = torch.std(tensor, dim=dim, keepdim=True)
    normalized_tensor = (tensor - mean) / (std + eps)
    return normalized_tensor


def minmax_normalize(tensor, dim=(-2, -1), eps=1e-8):
    min_val = torch.amin(tensor, dim=dim, keepdim=True)
    max_val = torch.amax(tensor, dim=dim, keepdim=True)
    normalized_tensor = (tensor - min_val) / (max_val - min_val + eps)
    return normalized_tensor


def rgb_to_ycbcr(rgb_img):
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


def ycbcr_to_rgb(ycbcr_img):
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


def chroma_downsampling(img_ycbcr, **kwargs):
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


def pad_image(img, patch_size, *args, **kwargs):
    """
    Pads an image tensor to ensure it can be evenly divided into patches of size (P, Q).

    Args:
    - img (torch.Tensor): The input image tensor of shape (C, H, W).
    - P (int): The height of each patch.
    - Q (int): The width of each patch.
    - padding_mode (str): The type of padding ('constant', 'reflect', etc.).
    - padding_value (int or float): The value used for constant padding.

    Returns:
    - torch.Tensor: The padded image tensor.
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


def unpad_image(padded_img, orig_size):
    """
    Recover the original image from a padded image by removing the added padding.

    Args:
    - orig_size (torch.Tensor): The padded image tensor of shape (C, H', W').
    - orig_size (tuple[int, int]): The original size of the image before padding.

    Returns:
    - torch.Tensor: The original image tensor of shape (C, H_orig, W_orig).
    """

    _, H_padded, W_padded = padded_img.shape
    H_orig, W_orig = orig_size
    start_h = (H_padded - H_orig) // 2
    end_h = start_h + H_orig
    start_w = (W_padded - W_orig) // 2
    end_w = start_w + W_orig
    original_img = padded_img[:, start_h:end_h, start_w:end_w]
    return original_img


def to_dtype(tensor, dtype):
    """
    Converts a tensor to the specified dtype, clamping its values to be within
    the representable range of the target dtype.

    Parameters:
    - tensor: A PyTorch tensor.
    - dtype: The target dtype to convert the tensor to.

    Returns:
    - A tensor of the target dtype with its values clamped within the representable range.
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


def quantize(tensor, target_dtype):
    """
    Quantize a float tensor to a specified integer data type.

    Args:
    - tensor (torch.Tensor): Input tensor of type torch.float32.
    - target_dtype (torch.dtype): Target integer data type (e.g., torch.uint8, torch.int16).

    Returns:
    - torch.Tensor: Quantized tensor.
    - float: Scale factor used for quantization.
    - float: Minimum value of the input tensor.
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


def dequantize(quantized_tensor, scale, min_val):
    """
    Dequantize a tensor to float32 values.

    Args:
    - quantized_tensor (torch.Tensor): Input quantized tensor.
    - scale (float): Scale factor used for quantization.
    - min_val (float): Minimum value of the original tensor before quantization.
    - target_dtype (torch.dtype): Target data type for the dequantized tensor.

    Returns:
    - torch.Tensor: Dequantized tensor.
    """

    # Convert to float32 for calculation
    quantized_tensor = quantized_tensor.to(torch.float32)

    # Dequantize
    dequantized = (quantized_tensor - quantized_tensor.min()) * scale + min_val

    return dequantized


def get_memory_usage(obj):
    """
    Calculates the memory usage of an object containing NumPy arrays or PyTorch tensors in bytes.

    Args:
    - obj: The data structure.

    Returns:
    - int: The memory usage of the data structure in bytes.
    """

    if isinstance(obj, (list, tuple, set)):
        return sum(get_memory_usage(item) for item in obj)
    elif isinstance(obj, dict):
        return sum(get_memory_usage(item) for item in obj.values())
    elif isinstance(obj, bytes):
        return len(obj)
    elif isinstance(obj, np.ndarray):
        # For NumPy array, use the nbytes attribute and convert to bits
        return obj.nbytes
    elif isinstance(obj, torch.Tensor):
        # For PyTorch tensor, calculate as before
        return obj.numel() * obj.element_size()
    else:
        raise ValueError(
            "Unsupported data type. Please provide an object containing NumPy arrays or PyTorch tensors."
        )


def get_compression_ratio(input, compressed):
    input_memory = get_memory_usage(input)
    compressed_memory = get_memory_usage(compressed)
    return input_memory / compressed_memory


def get_bbp(size, compressed):
    num_pixels = prod(size)
    compressed_memory = get_memory_usage(compressed)
    return compressed_memory * 8 / num_pixels


def _combine_bytes(payload1: bytes, payload2: bytes) -> bytes:
    """
    Encodes two bytes objects into a single bytes object.

    Args:
    - payload1: First bytes object to encode.
    - payload2: Second bytes object to encode.

    Returns:
    - A single combined bytes object containing both payloads.
    """

    if not isinstance(payload1, bytes) or not isinstance(payload2, bytes):
        raise TypeError("Both payload1 and payload2 must be bytes objects.")

    # Ensuring the length of payload1 can be represented in 4 bytes (up to 4GB for payload1)
    if len(payload1) > 0xFFFFFFFF:
        raise ValueError("payload1 is too large to encode.")

    len_payload1 = len(payload1).to_bytes(4, byteorder="big")
    return len_payload1 + payload1 + payload2


def _separate_bytes(combined: bytes) -> tuple[bytes, bytes]:
    """
    Decodes the combined bytes object back into two original bytes objects.

    Args:
    - combined: The combined bytes object containing both payloads.

    Return:
    - A tuple containing the two original bytes objects (payload1, payload2).
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
    return functools.reduce(_combine_bytes, payloads)


def separate_bytes(combined: bytes, num_payloads: int = 2) -> tuple[bytes, ...]:
    """
    Splits the combined bytes object into its parts (payloads).

    Args:
    - combined: The combined bytes object containing multiple payloads.
    - num_payloads: The number of payloads.

    Return:
    - A tuple containing the original bytes objects (payloads).
    """

    payloads = []
    payload1 = combined
    for _ in range(num_payloads - 1):
        payload1, payload2 = _separate_bytes(payload1)
        payloads.insert(0, payload2)

    payloads.insert(0, payload1)
    return tuple(payloads)


def dict_to_bytes(dictionary: dict) -> bytes:
    """Encode a dictionary into bytes."""
    json_string = json.dumps(dictionary)
    encoded_bytes = json_string.encode("utf-8")
    return encoded_bytes


def bytes_to_dict(encoded_bytes: bytes) -> dict:
    """Decode bytes back into a dictionary."""
    json_string = encoded_bytes.decode("utf-8")
    dictionary = json.loads(json_string)
    return dictionary


def encode_matrix(matrix: Tensor, mode: str = "col") -> bytes:
    assert len(matrix.shape) == 2, "'matrix' must be a 2D tensor."
    assert mode in {"col", "row"}, "'mode' must be either 'col' or 'row'."

    if mode == "col":
        fibers = matrix.split(split_size=1, dim=1)

    else:  # row
        fibers = matrix.split(split_size=1, dim=0)

    encoded_fibers = []
    for fiber in fibers:
        encoded_fiber = fiber.numpy().tobytes()
        encoded_fiber = zlib.compress(encoded_fiber)
        encoded_fibers.append(encoded_fiber)

    metadata = {
        "num_fibers": len(fibers),
        "mode": mode,
        "dtype": str(matrix.dtype).split(".")[-1],
    }
    encoded_metadata = dict_to_bytes(metadata)

    encoded_fibers = combine_bytes(encoded_fibers)
    encodeld_matrix = combine_bytes([encoded_metadata, encoded_fibers])

    return encodeld_matrix


def decode_matrix(encodeld_matrix: bytes, mode: str = "col") -> Tensor:
    assert mode in {"col", "row"}, "'mode' must be either 'col' or 'row'."

    encoded_metadata, encoded_fibers = separate_bytes(encodeld_matrix)

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


def encode_tensor(tensor: Tensor, *args, **kwargs) -> bytes:
    """
    Encode a PyTorch tensor and encode its shape and dtype using Zstandard algorithm,
    returning a single bytes object containing both the encoded tensor data and its metadata.

    Parameters:
    tensor (torch.Tensor): The tensor to encode.

    Returns:
    bytes: A single bytes object containing both encoded data and metadata.
    """
    if tensor.ndim == 2:
        return encode_matrix(tensor, *args, **kwargs)

    # Convert the tensor to bytes
    encoded_array = tensor.numpy().tobytes()

    # Encode the bytes using a lossless compression
    encoded_array = zlib.compress(encoded_array)

    # Prepare metadata
    metadata = {"shape": tensor.shape, "dtype": str(tensor.dtype).split(".")[-1]}
    encoded_metadata = dict_to_bytes(metadata)

    # Combine metadata and tensor data into a single bytes object
    encoded_tensor = combine_bytes([encoded_metadata, encoded_array])
    return encoded_tensor


def decode_tensor(encoded_tensor: bytes, *args, **kwargs) -> Tensor:
    """
    Decode a combined bytes object back into a PyTorch tensor using the encoded shape and dtype.

    Parameters:
    encoded_tensor (bytes): The combined bytes object containing both encoded data and metadata.

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


def encode_model(model, compress="zlib", **kwargs):
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=compress, **kwargs)
    return buffer.getvalue()


def decode_model(encodel_model, *args, **kwargs):
    buffer = io.BytesIO(encodel_model)
    model = joblib.load(buffer)
    return model
