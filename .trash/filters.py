def butterworth_filter(size, order, cutoff, device="cpu"):
    """
    Create a 2D Butterworth lowpass filter using PyTorch.

    Parameters:
    - size: tuple of ints, the size of the filter (height, width).
    - order: int, the order of the filter.
    - cutoff: float, the cutoff frequency (distance in the frequency domain).
    - device: str, the device to create the filter on ('cpu' or 'cuda').

    Returns:
    - A 2D Butterworth lowpass filter as a PyTorch tensor.
    """
    height, width = size
    # Create a meshgrid of u and v values
    u = torch.arange(height, device=device)
    v = torch.arange(width, device=device)
    # Calculate the Butterworth filter
    c_u, c_v = cutoff
    filter_u = 1 / (1 + torch.tan((math.pi / 2) * u / c_u) ** (2 * order))
    filter_v = 1 / (1 + torch.tan((math.pi / 2) * v / c_v) ** (2 * order))
    return torch.outer(filter_u, filter_v)


def cosine_filter(size, device="cpu"):
    height, width = size
    # Create a meshgrid of u and v values
    u = torch.arange(height, device=device)
    v = torch.arange(width, device=device)
    filter_u = (1 + torch.cos(math.pi * u / height)) / 2
    filter_v = (1 + torch.cos(math.pi * v / height)) / 2
    return torch.outer(filter_u, filter_v)
