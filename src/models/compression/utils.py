import torch


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
