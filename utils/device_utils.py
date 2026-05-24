import torch

def get_device(local_rank=None):

    if local_rank is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")