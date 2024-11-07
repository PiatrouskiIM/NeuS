import torch


def position_encoding(x: torch.Tensor, n: int):
    x = x.unsqueeze(0) * (1 << torch.arange(end=n, device=x.device)).unsqueeze(1)
    return torch.stack((torch.sin(x), torch.cos(x)), dim=1)
