import numpy as np
import torch
from skimage.metrics import structural_similarity


def mae(tensor1, tensor2):
    return torch.mean(torch.abs(tensor1 - tensor2), dim=(-3, -2, -1))


def mse(tensor1, tensor2):
    return torch.mean((tensor1 - tensor2) ** 2, dim=(-3, -2, -1))


def relative_error(tensor1, tensor2, epsilon: float = 1e-16):
    numerator = torch.norm(tensor1 - tensor2, p=2, dim=(-3, -2, -1))
    denominator = torch.norm(tensor1, p=2, dim=(-3, -2, -1))
    return numerator / (denominator + epsilon)


def psnr(tensor1, tensor2):
    mse_value = mse(tensor1, tensor2)
    max_pixel = torch.amax(tensor1, dim=(-3, -2, -1))
    psnr_value = 20 * torch.log10(max_pixel / torch.sqrt(mse_value))
    return psnr_value


def ssim(tensor1, tensor2):
    """
    Compute the Structural Similarity Index Measure (SSIM) between two PyTorch tensors
    over the last three dimensions (C, H, W).

    Parameters:
    - tensor1, tensor2: PyTorch tensors of shape (*, C, H, W) where
      * denotes any number of leading dimensions,
      C is the number of channels (e.g., 3 for RGB),
      H is the height,
      W is the width.

    Returns:
    - SSIM index: A float value representing the structural similarity over the last three dimensions.
    """
    # Ensure the input tensors are on CPU and convert them to numpy arrays
    tensor1 = tensor1.cpu().numpy()
    tensor2 = tensor2.cpu().numpy()

    # Initialize a list to store SSIM values
    ssim_values = []

    # Compute SSIM for each element in the tensor, considering all leading dimensions as a flattened list
    batch_shape = tensor1.shape[:-3]
    spatial_shape = tensor1.shape[-3:]
    flat_shape = (-1, *spatial_shape)
    tensor1_flat = tensor1.reshape(flat_shape)
    tensor2_flat = tensor2.reshape(flat_shape)

    for i in range(tensor1_flat.shape[0]):
        img1, img2 = tensor1_flat[i], tensor2_flat[i]

        # Compute data range
        data_range = img1.max() - img1.min()

        # Compute SSIM, note:
        ssim_index = structural_similarity(
            img1, img2, channel_axis=0, data_range=data_range
        )
        ssim_values.append(ssim_index)

    # Stack ssim values into an array
    ssim_values = np.stack(ssim_values)

    # Reshape back ssim value
    ssim_values = ssim_values.reshape(batch_shape)

    return torch.from_numpy(ssim_values)
