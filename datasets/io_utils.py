from PIL import Image
import torch
import numpy as np

# Utils used in dataset - DetectionDataset & RecognitionDataset

def read_img_as_tensor(path: str, dtype=torch.float32) -> torch.Tensor:
    """Reads an image file and converts it to a normalized torch tensor."""
    image = Image.open(path).convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
    tensor = torch.from_numpy(image).permute(2, 0, 1).type(dtype)  # HWC to CHW
    return tensor

def tensor_from_numpy(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    """Converts a NumPy array (HWC or CHW) to a torch tensor, normalized."""
    if arr.ndim == 3 and arr.shape[2] in [1, 3]:  # HWC
        arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.from_numpy(arr).type(dtype)
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    return tensor

def _copy_tensor(x: torch.Tensor) -> torch.Tensor:
    """Safely clone and detach a tensor to avoid gradient tracking."""
    return x.clone().detach()