import torch


def get_device(device):
    """
    Returns the best available device if unspecified: CUDA, MPS, or CPU
    """
    # Check if CUDA is available
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")

        # Check for Apple's Metal Performance Shaders (MPS) for M1/M2 Macs
        elif "mps" in torch.backends.__dict__ and torch.backends.mps.is_available():
            return torch.device("mps")

        else:
            return torch.device("cpu")

    # Default to CPU
    else:
        return torch.device(device)
