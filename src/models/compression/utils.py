import torch


def zscore_normalize(tensor):
    mean = torch.mean(dim=(-2, -1), keepdim=True)
    std = torch.std(dim=(-2, -1), keepdim=True)
    normalized_tensor = (tensor - mean) / (std + 1e-8)
    return normalized_tensor


def minmax_normalize(tensor):
    min_val = torch.amin(tensor, dim=(-3, -2, -1), keepdim=True)
    max_val = torch.amax(tensor, dim=(-3, -2, -1), keepdim=True)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor
