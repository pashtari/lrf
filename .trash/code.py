import torch
import numpy as np


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


def sin_positional_embeddings(channels, height, width):
    """Calculates 2D sinusoidal positional embeddings."""

    assert channels % 4 == 0, "Number of channels must be divisible by 4."

    # Frequency generation for sinusoidal embedding
    freqs = torch.exp(torch.arange(0, channels, 4) * (-math.log(10000.0) / channels))
    num_freqs = channels // 4
    freqs = freqs.reshape(-1, 1, 1).expand(num_freqs, height, width)

    # Positional grids
    pos_y = torch.arange(height).reshape(1, -1, 1).expand(num_freqs, height, width)
    pos_x = torch.arange(width).reshape(1, 1, -1).expand(num_freqs, height, width)

    # Compute the sinusoidal embeddings
    cos_y = torch.cos(freqs * pos_y)
    sin_y = torch.sin(freqs * pos_y)
    cos_x = torch.cos(freqs * pos_x)
    sin_x = torch.sin(freqs * pos_x)

    # Concatenate along the channel dimension
    position_embeddings = torch.cat((cos_y, sin_y, cos_x, sin_x), dim=0)

    return position_embeddings


def coordinate_channels(shape):
    assert len(shape) == 2
    H, W = shape

    y = torch.linspace(0, 1, steps=H)
    x = torch.linspace(0, 1, steps=W)

    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

    coordinates = torch.stack((grid_y, grid_x))

    return coordinates
