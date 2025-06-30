import json
import os
from typing import Any, Callable
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from copy import deepcopy

from doctane.datasets.transform_utils import get_img_shape, pre_transform_multiclass
from doctane.datasets.io_utils import _copy_tensor, read_img_as_tensor, tensor_from_numpy


CLASS_NAME: str = ["row", "column"]


class DetectionDataset:
    """
    Custom dataset for text detection tasks.

    Supports polygon annotations (multi-class or single-class),
    preprocessing, and transformations using albumentations.
    """

    def __init__(
        self,
        img_folder: str | Path,
        label_path: str,
        img_transforms: Callable[[Any], Any] | None = None,
        sample_transforms: Callable[[Any, Any], tuple[Any, Any]] | None = None,
        use_polygons: bool = False,
    ) -> None:
        self.root = str(img_folder)
        if not Path(self.root).is_dir():
            raise ValueError(f"Invalid image folder path: {img_folder}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        self.img_transforms = img_transforms
        self.sample_transforms = sample_transforms
        self._pre_transforms = pre_transform_multiclass
        self._get_img_shape = get_img_shape
        self._class_names: list[str] = []
        self.data: list[tuple[str, tuple[np.ndarray, list[str]]]] = []

        # Load and process label file
        with open(label_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

        for img_name, label in labels.items():
            img_path = os.path.join(self.root, img_name)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            
            geoms, class_names = self.format_polygons(
                label["polygons"], use_polygons=use_polygons, np_dtype=np.float32
            )
            self.data.append((img_name, (geoms, class_names)))
            self._class_names.extend(class_names)

        # Deduplicate class names
        self._class_names = sorted(set(self._class_names))

    def format_polygons(
        self,
        polygons: list | dict,
        use_polygons: bool,
        np_dtype: type,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Converts polygon annotations into bounding boxes or polygon format.
        Supports single-class (list) and multi-class (dict) format.
        """
        if isinstance(polygons, list):
            polygons_classes = [CLASS_NAME] * len(polygons)
            _polygons = np.asarray(polygons, dtype=np_dtype)

        elif isinstance(polygons, dict):
            polygons_classes = [k for k, v in polygons.items() for _ in v]
            _polygons = np.concatenate(
                [np.asarray(poly, dtype=np_dtype) for poly in polygons.values() if poly],
                axis=0
            ) if any(polygons.values()) else np.zeros((0, 4, 2), dtype=np_dtype)

        else:
            raise TypeError(f"Expected list or dict for polygons, got {type(polygons)}")

        if use_polygons:
            geoms = _polygons
        else:
            geoms = np.concatenate((_polygons.min(axis=1), _polygons.max(axis=1)), axis=1)

        return geoms, polygons_classes

    def __len__(self) -> int:
        return len(self.data)

    def _read_sample(self, index: int) -> tuple[torch.Tensor, Any]:
        """Loads an image and its corresponding annotations."""
        img_name, target = self.data[index]

        # Load image tensor
        img = (
            tensor_from_numpy(img_name)
            if isinstance(img_name, np.ndarray)
            else read_img_as_tensor(os.path.join(self.root, img_name))
        )

        return img, deepcopy(target)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Applies all transformations and returns a sample."""
        img, target = self._read_sample(index)

        if self._pre_transforms:
            img, target = self._pre_transforms(img, target)

        if self.img_transforms:
            img = self.img_transforms(img)

        if self.sample_transforms is not None:
            # Conditions to assess it is detection model with multiple classes and avoid confusion with other tasks.
            if (
                isinstance(target, dict)
                and all(isinstance(item, np.ndarray) for item in target.values())
                and set(target.keys()) != {"boxes", "labels"}  # avoid confusion with obj detection target
            ):
                img_transformed = _copy_tensor(img)
                for class_name, bboxes in target.items():
                    img_transformed, target[class_name] = self.sample_transforms(img, bboxes)
                img = img_transformed
            else:
                img, target = self.sample_transforms(img, target)

        return img, target

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"

    @property
    def class_names(self) -> list[str]:
        """Returns unique class names."""
        return self._class_names

    @staticmethod
    def collate_fn(samples: list[tuple[torch.Tensor, Any]]) -> tuple[torch.Tensor, list[Any]]:
        """Collates samples into a batch."""
        images, targets = zip(*samples)
        images = torch.stack(images, dim=0)
        return images, list(targets)
