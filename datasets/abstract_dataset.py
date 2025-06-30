import os
from pathlib import Path
from typing import Callable, Any

import numpy as np
import torch
from copy import deepcopy

from doctane.datasets.utils import get_img_shape, tensor_from_numpy, read_img_as_tensor, _copy_tensor, download_from_url


__all__ = ["AbstractDataset", "VisionDataset"]


class AbstractDataset:
    """
    A base abstract dataset class for handling image and target loading,
    as well as applying various transformations (pre, image-level, and sample-level).
    """

    data: list[Any] = []  # Should be populated externally with (img_name, target) tuples
    _pre_transforms: Callable[[Any, Any], tuple[Any, Any]] | None = None

    def __init__(
        self,
        root: str | Path,
        img_transforms: Callable[[Any], Any] | None = None,
        sample_transforms: Callable[[Any, Any], tuple[Any, Any]] | None = None,
        pre_transforms: Callable[[Any, Any], tuple[Any, Any]] | None = None,
    ) -> None:
        if not Path(root).is_dir():
            raise ValueError(f"Expected a path to a reachable folder: {root}")
        
        self.root = Path(root)
        self.img_transforms = img_transforms
        self.sample_transforms = sample_transforms
        self._pre_transforms = pre_transforms
        self._get_img_shape = get_img_shape

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """
        Retrieves and processes a sample from the dataset.
        Applies pre-processing, image-level transforms, and sample-level transforms.
        """
        img, target = self._read_sample(index)

        # Pre-processing step
        if self._pre_transforms is not None:
            img, target = self._pre_transforms(img, target)

        # Apply image-level transforms
        if self.img_transforms is not None:
            img = self.img_transforms(img)

        # Apply sample-level transforms (may modify image and/or target)
        if self.sample_transforms is not None:
            if (
                isinstance(target, dict)
                and all(isinstance(item, np.ndarray) for item in target.values())
                and set(target.keys()) != {"boxes", "labels"}  # Avoid confusion with object detection format
            ):
                img_transformed = _copy_tensor(img)
                for class_name, bboxes in target.items():
                    img_transformed, target[class_name] = self.sample_transforms(img, bboxes)
                img = img_transformed
            else:
                img, target = self.sample_transforms(img, target)

        return img, target

    def _read_sample(self, index: int) -> tuple[torch.Tensor, Any]:
        """
        Reads a sample's image and target from the dataset.
        Converts image to tensor and deepcopies the target to prevent mutation.
        """
        img_name, target = self.data[index]

        # Validate target structure
        if isinstance(target, dict):
            assert "boxes" in target, "Target dictionary must contain 'boxes'"
            assert "labels" in target, "Target dictionary must contain 'labels'"
        elif isinstance(target, tuple):
            assert len(target) == 2, "Tuple target must have 2 elements"
            assert isinstance(target[0], (str, np.ndarray)), "First tuple element must be string or numpy array"
            assert isinstance(target[1], list), "Second tuple element must be a list"
        else:
            assert isinstance(target, (str, np.ndarray)), "Target must be string or numpy array"

        # Load image as tensor
        if isinstance(img_name, np.ndarray):
            img = tensor_from_numpy(img_name, dtype=torch.float32)
        else:
            img_path = os.path.join(self.root, img_name)
            img = read_img_as_tensor(img_path, dtype=torch.float32)

        return img, deepcopy(target)

    def extra_repr(self) -> str:
        """Optional string representation for printing dataset info."""
        return ""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"

    @staticmethod
    def collate_fn(samples: list[tuple[torch.Tensor, Any]]) -> tuple[torch.Tensor, list[Any]]:
        """
        Custom collate function for use with PyTorch DataLoader.
        Batches images and preserves original target structure.
        """
        images, targets = zip(*samples)
        images = torch.stack(images, dim=0)
        return images, list(targets)



class VisionDataset(AbstractDataset):
    """
    Dataset class with support for downloading and extracting datasets.

    Args:
        url: Download URL for the dataset
        file_name: Optional name of the downloaded file
        file_hash: Optional SHA256 hash for verification
        extract_archive: Whether the downloaded file should be extracted
        download: Whether to download if file is not found
        overwrite: Force re-extraction even if folder exists
        cache_dir: Optional cache directory path
        cache_subdir: Subdirectory under the cache directory
    """

    def __init__(
        self,
        url: str,
        file_name: str | None = None,
        file_hash: str | None = None,
        extract_archive: bool = False,
        download: bool = False,
        overwrite: bool = False,
        cache_dir: str | None = None,
        cache_subdir: str | None = None,
        **kwargs: Any,
    ) -> None:
        cache_dir = cache_dir or os.environ.get("DOCTANE_CACHE_DIR", os.path.join(Path.home(), ".cache_datasets", "doctane"))
        cache_subdir = cache_subdir or "datasets"
        file_name = file_name or os.path.basename(url)

        archive_path = Path(os.path.join(cache_dir, cache_subdir, file_name))

        if not archive_path.exists() and not download:
            raise FileNotFoundError(f"{archive_path} not found. Set `download=True` to fetch it.")

        archive_path = Path(download_from_url(url, file_name, file_hash, cache_dir, cache_subdir))

        dataset_path = archive_path
        if extract_archive:
            dataset_path = archive_path.parent / archive_path.stem
            if overwrite or not dataset_path.is_dir():
                shutil.unpack_archive(archive_path, dataset_path)

        super().__init__(dataset_path, **kwargs)
