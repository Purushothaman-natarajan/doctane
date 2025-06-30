import json
import os
from pathlib import Path
from typing import Any, Callable
from copy import deepcopy

import numpy as np
import torch

from doctane.datasets.transform_utils import get_img_shape
from doctane.datasets.io_utils import _copy_tensor, read_img_as_tensor, tensor_from_numpy

class RecognitionDataset:
    """
    Dataset class for single-class text recognition tasks.
    
    Each image is associated with a single label (string or numpy array).
    
    Args:
        img_folder (str | Path): Directory with images.
        labels_path (str): Path to JSON file mapping image names to labels.
        img_transforms (Callable, optional): Image-only transform function.
        sample_transforms (Callable, optional): Transform function applied to (image, label) pairs.
        pre_transforms (Callable, optional): Transform function applied before other transforms.
    """

    def __init__(
        self,
        img_folder: str | Path,
        labels_path: str,
        img_transforms: Callable[[Any], Any] | None = None,
        sample_transforms: Callable[[Any, Any], tuple[Any, Any]] | None = None,
        pre_transforms: Callable[[Any, Any], tuple[Any, Any]] | None = None,
    ) -> None:
        self.root = Path(img_folder)
        if not self.root.is_dir():
            raise ValueError(f"Invalid image folder path: {img_folder}")

        self.img_transforms = img_transforms
        self.sample_transforms = sample_transforms
        self._pre_transforms = pre_transforms
        self._get_img_shape = get_img_shape
        self.data: list[tuple[str, Any]] = []

        # Load and validate label file
        with open(labels_path, encoding="utf-8") as f:
            labels = json.load(f)

        for img_name, label in labels.items():
            img_path = self.root / img_name
            if not img_path.exists():
                raise FileNotFoundError(f"Missing image: {img_path}")
            self.data.append((img_name, label))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Any]:
        img_name, target = self.data[index]

        # Validate target for single-class recognition
        assert isinstance(target, (str, np.ndarray)), "Target must be a string or a numpy array"

        # Load image as torch tensor
        img = (
            tensor_from_numpy(img_name, dtype=torch.float32)
            if isinstance(img_name, np.ndarray)
            else read_img_as_tensor(self.root / img_name, dtype=torch.float32)
        )

        target = deepcopy(target)

        # Apply pre-transforms
        if self._pre_transforms:
            img, target = self._pre_transforms(img, target)

        # Apply image-only transforms
        if self.img_transforms:
            img = self.img_transforms(img)

        # Apply sample transforms
        if self.sample_transforms:
            img, target = self.sample_transforms(img, target)

        return img, target

    @staticmethod
    def collate_fn(samples: list[tuple[torch.Tensor, Any]]) -> tuple[torch.Tensor, list[Any]]:
        """
        Collate function for DataLoader — batches images and returns list of targets.
        """
        images, targets = zip(*samples)
        return torch.stack(images, dim=0), list(targets)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_samples={len(self)}, root={self.root})"
