from functools import reduce
from operator import mul
import json
import zlib

import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np


def prod(x):
    return reduce(mul, x, 1)


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
    - orig_size (Tuple[int, int]): The original size of the image before padding.

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


def pack_tensor(tensor, min_value):
    # Convert to numpy array
    numpy_array = tensor.clone().numpy().astype(np.int32)

    # Add 16 to all elements to make them positive
    numpy_array -= min_value

    # Convert to uint8
    numpy_array = numpy_array.astype(np.uint8)

    # Convert to binary array
    binary_array = np.unpackbits(numpy_array).reshape(-1, 8)

    # Pick the last 5 bits
    binary_array = binary_array[..., -5:]

    # Pack the elements into bits in a uint8 array
    packed_array = np.packbits(binary_array)

    return torch.from_numpy(packed_array)


def unpack_tensor(packed_tensor, shape, min_value, dtype):
    # Convert to numpy array
    packed_array = packed_tensor.clone().numpy()

    # Unpack the binary array
    binary_array = np.unpackbits(packed_array)

    # Unpad and reshape binary array
    numel = prod(shape)
    num_bits_per_elements = binary_array.size // prod(shape)
    num_bits = numel * num_bits_per_elements
    binary_array = binary_array[:num_bits].reshape(-1, num_bits_per_elements)

    # Zero pad, prepend
    pad_width = 8 - num_bits_per_elements
    binary_array = np.pad(binary_array, pad_width=((0, 0), (pad_width, 0)))

    # Pack the elements into bits in a uint8 array
    numpy_array = np.packbits(binary_array, axis=-1).reshape(*shape).astype(np.int32)

    # Subtract 16 from all elements to restore the original values
    numpy_array += min_value

    return torch.from_numpy(numpy_array).to(dtype)


def combine_bytes(payload1: bytes, payload2: bytes) -> bytes:
    """
    Encodes two bytes objects into a single bytes object.

    :param payload1: First bytes object to encode.
    :param payload2: Second bytes object to encode.
    :return: A single combined bytes object containing both payloads.
    """
    if not isinstance(payload1, bytes) or not isinstance(payload2, bytes):
        raise TypeError("Both payload1 and payload2 must be bytes objects.")

    # Ensuring the length of payload1 can be represented in 4 bytes (up to 4GB for payload1)
    if len(payload1) > 0xFFFFFFFF:
        raise ValueError("payload1 is too large to encode.")

    len_payload1 = len(payload1).to_bytes(4, byteorder="big")
    return len_payload1 + payload1 + payload2


def separate_bytes(combined: bytes) -> (bytes, bytes):
    """
    Decodes the combined bytes object back into two original bytes objects.

    :param combined: The combined bytes object containing both payloads.
    :return: A tuple containing the two original bytes objects (payload1, payload2).
    """
    if not isinstance(combined, bytes):
        raise TypeError("Combined must be a bytes object.")

    if len(combined) < 4:
        raise ValueError("Combined data is too short to decode.")

    len_payload1 = int.from_bytes(combined[:4], byteorder="big")
    payload1 = combined[4 : 4 + len_payload1]
    payload2 = combined[4 + len_payload1 :]
    return payload1, payload2


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


def encode_tensor(tensor):
    """
    Encode a PyTorch tensor and encode its shape and dtype using Zstandard algorithm,
    returning a single bytes object containing both the encoded tensor data and its metadata.

    Parameters:
    tensor (torch.Tensor): The tensor to encode.

    Returns:
    bytes: A single bytes object containing both encoded data and metadata.
    """
    # Convert the tensor to bytes
    array_bytes = tensor.numpy().tobytes()

    # Encode the bytes using a lossless compression
    array_bytes = zlib.compress(array_bytes)

    # Prepare metadata
    metadata = {"shape": tensor.shape, "dtype": str(tensor.dtype).split(".")[-1]}
    metadata_bytes = dict_to_bytes(metadata)

    # Combine metadata and tensor data into a single bytes object
    encoded_tensor = combine_bytes(metadata_bytes, array_bytes)
    return encoded_tensor


def decode_tensor(encoded_tensor):
    """
    Decode a combined bytes object back into a PyTorch tensor using the encoded shape and dtype.

    Parameters:
    encoded_tensor (bytes): The combined bytes object containing both encoded data and metadata.

    Returns:
    torch.Tensor: The decoded tensor.
    """
    # Extract metadata and array data
    metadata_bytes, array_bytes = separate_bytes(encoded_tensor)

    # Decode metadata
    metadata = bytes_to_dict(metadata_bytes)
    shape = metadata["shape"]
    dtype = metadata["dtype"]

    # Decode the array data
    decoded_array = zlib.decompress(array_bytes)

    # Convert back to tensor
    decoded_tensor = torch.from_numpy(
        np.frombuffer(decoded_array, dtype=np.dtype(dtype)).reshape(shape)
    )

    return decoded_tensor


def encode_tensors(tensors):
    """
    Encode a sequence of PyTorch tensors, returning a single bytes object
    containing the encoded data and metadata for each tensor.

    Parameters:
    tensors (Sequence of torch.Tensor): The sequence of tensors to encode.

    Returns:
    bytes: A single bytes object containing all encoded tensors and their metadata.
    """
    encoded_parts = []

    for tensor in tensors:
        encoded_tensor = encode_tensor(tensor)
        part_length = len(encoded_tensor)
        encoded_parts.append(part_length.to_bytes(4, "big") + encoded_tensor)

    # Prefix the total number of tensors as 4 bytes
    total_tensors = len(tensors)
    all_data = total_tensors.to_bytes(4, "big") + b"".join(encoded_parts)

    return all_data


def decode_tensors(all_data):
    """
    Decode a bytes object back into a tuple of PyTorch tensors.

    Parameters:
    all_data (bytes): The bytes object containing all encoded tensors and their metadata.

    Returns:
    tuple: A tuple of decoded PyTorch tensors.
    """
    total_tensors = int.from_bytes(all_data[:4], "big")
    offset = 4
    tensors = []

    for _ in range(total_tensors):
        part_length = int.from_bytes(all_data[offset : offset + 4], "big")
        offset += 4  # Move past the length prefix
        tensor_data = all_data[offset : offset + part_length]
        tensor = decode_tensor(tensor_data)
        tensors.append(tensor)
        offset += part_length

    return tuple(tensors)