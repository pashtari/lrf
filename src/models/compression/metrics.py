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


def psnr(img1, img2, max_value=255):
    mse_value = mse(img1.to(torch.float32), img2.to(torch.float32))
    psnr_value = 20 * torch.log10(max_value / torch.sqrt(mse_value))
    return psnr_value


def ssim(img1, img2):
    """
    Compute the Structural Similarity Index Measure (SSIM) between two PyTorch tensors
    over the last three dimensions (C, H, W).

    Parameters:
    - img1, img2: PyTorch tensors of shape (C, H, W) where
      C is the number of channels (e.g., 3 for RGB),
      H is the height,
      W is the width.

    Returns:
    - SSIM index: A float value representing the structural similarity over the last three dimensions.
    """
    # Ensure the input tensors are on CPU and convert them to numpy arrays
    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()

    data_range = img1.max() - img1.min()

    ssim_value = structural_similarity(img1, img2, channel_axis=0, data_range=data_range)

    return torch.tensor(ssim_value)
