from PIL import Image
import requests
import os
from pathlib import Path
import torch
import numpy as np


# Simple Utility functions

def get_img_shape(img_path: str) -> tuple[int, int]:
    """
    Gets the shape (height, width) of an image file.
    Args:
        img_path (str): Path to the image file.

    Returns:
        tuple: Height and width of the image.
    """
    with Image.open(img_path) as img:
        return img.size[::-1]  # Returns (height, width)


def download_from_url(url: str, file_name: str, file_hash: str | None = None, cache_dir: str = None, cache_subdir: str = "datasets") -> str:
    """
    Downloads a file from a URL and stores it in the specified cache directory.
    Args:
        url (str): The URL to download from.
        file_name (str): The name to save the file as.
        file_hash (str, optional): Expected file hash for verification.
        cache_dir (str, optional): Directory to store the downloaded file.
        cache_subdir (str, optional): Subfolder under the cache directory.

    Returns:
        str: Path to the downloaded file.
    """
    cache_dir = Path(cache_dir or os.path.join(Path.home(), ".cache", "doctr", cache_subdir))
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_path = cache_dir / file_name

    # Download file if it doesn't exist
    if not file_path.exists():
        print(f"Downloading {url} to {file_path}...")
        response = requests.get(url)
        response.raise_for_status()

        with open(file_path, 'wb') as f:
            f.write(response.content)
    
    # Optionally, check file hash (not implemented here, but easy to add if needed)
    if file_hash:
        # You can use hashlib to verify the file hash here
        pass

    return str(file_path)


def _copy_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Returns a copy of the given tensor.
    Args:
        tensor (torch.Tensor): The tensor to copy.

    Returns:
        torch.Tensor: A new tensor that is a copy of the input.
    """
    return tensor.clone()


def tensor_from_numpy(array: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    """
    Converts a NumPy array image into a torch.Tensor.

    Args:
        array (np.ndarray): Image array in [H, W, C] format.
        dtype: Torch data type to cast the tensor to.

    Returns:
        torch.Tensor: Tensor in [C, H, W] format, scaled to [0, 1].
    """
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected image with shape [H, W, 3] for RGB.")

    tensor = torch.from_numpy(array.astype("float32") / 255.0).permute(2, 0, 1).to(dtype)
    return tensor


def read_img_as_tensor(img_path: str, dtype=torch.float32) -> torch.Tensor:
    """
    Reads an image from the given path and converts it into a torch.Tensor.

    Args:
        img_path (str): Path to the image file.
        dtype: Torch data type to cast the tensor to.

    Returns:
        torch.Tensor: Image as a torch tensor with shape [C, H, W], scaled to [0, 1].
    """
    with Image.open(img_path).convert("RGB") as img:
        img_np = np.array(img).astype("float32") / 255.0  # normalize to [0, 1]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(dtype)  # [H, W, C] -> [C, H, W]
        return img_tensor

