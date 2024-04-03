import os
from functools import reduce
from operator import mul
import json

import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms.v2.functional as FT
from PIL import Image
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import zstandard as zstd


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


def rgb_to_ycbcr(img):
    # Convert the PyTorch tensor to a PIL image
    img_rgb = FT.to_pil_image(img)
    # Convert the image to YCbCr color space
    img_ycbcr = img_rgb.convert("YCbCr")
    # Convert the PIL image to a PyTorch tensor
    img_ycbcr = FT.pil_to_tensor(img_ycbcr)
    return img_ycbcr


def ycbcr_to_rgb(img):
    # Convert the PyTorch tensor to a PIL image
    img_rgb = FT.to_pil_image(img)
    # Convert the image to RGB color space
    img_ycbcr = img_rgb.convert("RGB")
    # Convert the PIL image to a PyTorch tensor
    img_ycbcr = FT.pil_to_tensor(img_ycbcr)
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
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        info = torch.iinfo(dtype)
        dtype_min, dtype_max = info.min, info.max
    elif dtype in (torch.uint8,):
        info = torch.iinfo(dtype)
        dtype_min, dtype_max = info.min, info.max
    elif dtype in (torch.float32, torch.float64):
        info = torch.finfo(dtype)
        dtype_min, dtype_max = info.min, info.max
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Clamp the tensor values and convert to the target dtype
    clamped_tensor = torch.clamp(tensor, dtype_min, dtype_max).to(dtype)

    return clamped_tensor


def compute_bins(num_bins, num_samples=1000000):
    np.random.seed(42)

    x = np.random.normal(loc=0, scale=1, size=(num_samples, 1))
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    uniform_edges = np.linspace(x.min(), x.max(), num_bins + 1)
    init = (uniform_edges[1:] + uniform_edges[:-1]) * 0.5
    init = init.reshape(-1, 1)

    kmeans = KMeans(n_clusters=num_bins, init=init, n_init=1)
    kmeans.fit(x)

    bin_centers = kmeans.cluster_centers_.flatten()
    bin_centers.sort()
    bin_edges = (bin_centers[1:] + bin_centers[:-1]) * 0.5

    bin_centers = torch.from_numpy(bin_centers).to(torch.float32)
    bin_edges = torch.from_numpy(bin_edges).to(torch.float32)
    return bin_centers, bin_edges


_bin_centers, _ = compute_bins(num_bins=256)
_bin_centers_path = os.path.join(os.path.dirname(__file__), "bin_centers.pt")
torch.save(_bin_centers, _bin_centers_path)


def kbins_quantize(tensor, bin_centers=None):
    if bin_centers is None:
        bin_centers_path = os.path.join(os.path.dirname(__file__), "bin_centers.pt")
        bin_centers = torch.load(bin_centers_path)

    # Find mean and std values in the tensor
    mean, std = tensor.mean(), tensor.std()

    # Normalize the tensor
    tensor = (tensor - mean) / std

    # Quantize
    bin_edges = (bin_centers[1:] + bin_centers[:-1]) * 0.5
    quantized = torch.bucketize(tensor, bin_edges)

    return quantized.to(torch.uint8), mean.item(), std.item()


def kbins_dequantize(quantized_tensor, mean, std, bin_centers=None):
    if bin_centers is None:
        bin_centers_path = os.path.join(os.path.dirname(__file__), "bin_centers.pt")
        bin_centers = torch.load(bin_centers_path)

    # Quantize
    tensor = bin_centers[quantized_tensor.to(torch.int)]

    # Denormalize
    dequantized = std * tensor + mean

    return dequantized


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

    # Encode the bytes using Zstandard
    compressor = zstd.ZstdCompressor()
    array_bytes = compressor.compress(array_bytes)

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
    decompressor = zstd.ZstdDecompressor()
    decoded_array = decompressor.decompress(array_bytes)

    # Convert back to tensor
    decoded_tensor = torch.from_numpy(
        np.frombuffer(decoded_array, dtype=getattr(np, dtype)).reshape(shape)
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


# # Create a random tensor for testing
# original_tensor = (2 * torch.rand(784 * 16, 4) - 1).to(torch.int16)

# ###### tensor ######
# # Encode the tensor
# encoded_tensor = encode_tensor(original_tensor)

# # Decode the tensor
# decoded_tensor = decode_tensor(encoded_tensor)

# # Verify if the original tensor and decoded tensor are equal
# are_equal = torch.equal(original_tensor, decoded_tensor)

# # Compute the encodeion ratio
# original_size = original_tensor.element_size() * original_tensor.numel()
# encoded_size = len(encoded_tensor)
# encodeion_ratio = original_size / encoded_size

# print(f"Equal? {are_equal}")
# print(f"Original size: {original_size} bytes")
# print(f"Encoded size: {encoded_size} bytes")
# print(f"Encodeion ratio: {encodeion_ratio:.3f}")


# ###### Sequence of tensors ######
# # Example usage:
# tensor_tuple = (torch.randn(10, 10), torch.randn(5, 5, 5))

# # Encode the tuple of tensors
# encoded_data = encode_tensors(tensor_tuple)

# # Decode back into a tuple of tensors
# decoded_tensors = decode_tensors(encoded_data)

# # Verify if decoded tensors match the original tensors
# for original, decoded in zip(tensor_tuple, decoded_tensors):
#     print(f"Equal? {torch.equal(original, decoded)}")
