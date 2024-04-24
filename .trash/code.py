import functools
from operator import mul
import math
import io


import torch
from torch import Tensor
import numpy as np
from einops import rearrange
import joblib


def prod(x):
    return functools.reduce(mul, x, 1)


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


def encode_model(model, *args, **kwargs):
    buffer = io.BytesIO()
    joblib.dump(model, buffer, *args, **kwargs)
    return buffer.getvalue()


def decode_model(encodel_model, *args, **kwargs):
    buffer = io.BytesIO(encodel_model)
    model = joblib.load(buffer)
    return model


def to_zigzag_u(u: Tensor, size: tuple[int, int], patch_size: tuple[int, int]) -> Tensor:
    """Reorder the u factor matrix with zigzag pattern."""

    u = rearrange(u, "(h w) r -> r h w", h=size[0] // patch_size[0])
    u = zigzag_flatten(u).mT
    return u


def from_zigzag_u(
    u: Tensor, size: tuple[int, int], patch_size: tuple[int, int]
) -> Tensor:
    """Convert the zigzag-ordered factor matrix u back to the original one."""

    grid_size = size[0] // patch_size[0], size[1] // patch_size[1]
    u = zigzag_unflatten(u.mT, grid_size)
    u = rearrange(u, "r h w -> (h w) r")
    return u


def to_zigzag_v(v: Tensor, patch_size: tuple[int, int]) -> Tensor:
    """Reorder the v factor matrix with zigzag pattern."""

    p, q = patch_size
    v = rearrange(v, "(c p q) r -> (c r) p q", p=p, q=q)
    v = zigzag_flatten(v).mT
    return v


def from_zigzag_v(v: Tensor, patch_size: tuple[int, int], rank: int) -> Tensor:
    """Convert the zigzag-ordered factor matrix v back to the original one."""

    v = zigzag_unflatten(v.mT, patch_size)
    v = rearrange(v, "(c r) p q -> (c p q) r", r=rank)
    return v


def zigzag_flatten(x):
    assert x.dim() >= 2, "Input tensor must be at least 2D"
    M, N = x.shape[-2:]

    # Create an empty list to store the zigzag-flattened elements
    flattened = []

    # Iterate over the diagonals
    for i in range(M + N - 1):
        # Get the diagonal elements
        diag = x.flip(dims=[-2]).diagonal(offset=(i - M + 1), dim1=-2, dim2=-1)

        # Reverse the elements if the index is odd
        if i % 2 == 1:
            diag = diag.flip(dims=[-1])

        # Append the elements to the list
        flattened.append(diag)

    # Concatenate the elements in the list to get the flattened tensor
    flattened = torch.cat(flattened, dim=-1)

    return flattened


def zigzag_unflatten(x, shape):
    M, N = shape
    assert M * N == x.shape[-1]

    m, n = torch.meshgrid(torch.arange(M), torch.arange(N))

    # Create an empty tensor to store the unflattened elements
    unflattened = torch.empty_like(x).reshape(*x.shape[:-1], M, N)

    # Iterate over the diagonals
    start_diag = 0
    for i in range(M + N - 1):
        # Get the length of the diagonal
        length = min(i + 1, M + N - i - 1, M, N)

        diag = x[..., start_diag : start_diag + length].clone()
        start_diag += length

        if i % 2 == 0:
            diag = diag.flip(dims=[-1])

        print(diag)

        # # Fill the diagonal elements in the tensor
        unflattened[..., m + n == i] = diag

    return unflattened


# M, N = 4, 11
# # Create a 3x4 toy tensor
# x = torch.arange(3 * M * N).reshape(3, M, N)

# # Apply the zigzag_flatten function
# flattened_x = zigzag_flatten(x)

# # Print the original and flattened tensors
# print("Original tensor:")
# print(x)
# print("\nFlattened tensor:")
# print(flattened_x)


# # Test the function on a zigzag flattened tensor
# print("Original tensor:")
# x_hat = zigzag_unflatten(flattened_x, (M, N))
# print(x_hat)

# print("Is reconstructed tensor equel to the original one?")
# print(torch.equal(x_hat, x))


def apply(fun):
    """A decorator that applies a function to all elements in a list."""

    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        return list(map(fun, *args, **kwargs))

    return wrapper


@apply
def run_length_encode(x):
    if x.numel() == 0:
        return torch.empty(0, dtype=x.dtype), torch.empty(0, dtype=torch.int32)

    # Find places where the value changes
    diffs = x[1:] != x[:-1]
    # Find indices where the value changes
    indices = torch.where(diffs)[0] + 1
    # Append the final index for the last element
    indices = torch.cat((indices, torch.tensor([x.numel()], device=x.device)))
    # Calculate lengths
    lengths = torch.diff(torch.cat((torch.tensor([0], device=x.device), indices)))
    values = x[indices - lengths]

    return values, lengths


@apply
def run_length_decode(rle):
    values, lengths = rle
    return torch.repeat_interleave(values, lengths)


# # Example to test the implementation
# x = torch.tensor(
#     [[1, 1, 2, 2, 3], [4, 4, 4, 5, 5], [2, 7, 7, 7, 8]],
# )

# x_rle = run_length_encode(x)
# print("Encoded Values:", [*zip(*x_rle)][0])
# print("Encoded Counts:", [*zip(*x_rle)][1])

# x_hat = run_length_decode(x_rle)
# print("Decoded Tensor:", x_hat)

# # Verify if the original and decoded tensors are identical
# print(
#     "Is the original tensor equal to the decoded tensor?",
#     torch.equal(x, torch.stack(x_hat)),
# )


from torchvision.transforms import v2
import matplotlib.pyplot as plt
from skimage import data
from skimage.io import imread

import lrf

# Load the astronaut image
# image = data.astronaut()
image = imread("./data/kodak/kodim15.png")


# transforms = v2.Compose([v2.ToImage(), v2.Resize(size=(224, 224), interpolation=2)])
transforms = v2.Compose([v2.ToImage()])


# Transform the input image
image = torch.tensor(transforms(image))


def mask_image(x, portion=0.1):
    step = max(round(1 / portion), 1)
    mask = torch.zeros_like(x[0]).flatten()
    mask[::step] = 1
    x_masked = []
    for c in range(3):
        channel_masked = zigzag_flatten(x[c]) * mask
        x_masked.append(channel_masked)

    x_masked = run_length_encode(x_masked)
    # x_masked = torch.stack(x_masked)
    return x_masked


masked_image = mask_image(image, 0.01)

compressed_size = sum(
    len(lrf.encode_matrix(masked_image[i][j].reshape(-1, 1)))
    for i in range(3)
    for j in range(2)
)
# compressed_size = len(lrf.encode_tensor(masked_image))

print(f"bpp: {compressed_size*8/prod(image.shape[-2:])}")

# Visualize image
plt.imshow(image.permute(1, 2, 0))
plt.axis("off")
plt.title("Original Image")
plt.show()
