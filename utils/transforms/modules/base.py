
# Image and target transformation utilities 


import math
import random
from collections.abc import Callable
from typing import Any

import numpy as np

from doctane.utils.representation import NestedObject
from doctane.utils.transforms import functional as F

__all__ = ["SampleCompose", "ImageTransform", "ColorInversion", "OneOf", "RandomApply", "RandomRotate", "RandomCrop"]


class SampleCompose(NestedObject):
    """
    Applies a sequence of transformations to both image and target.

    Args:
        transforms: List of transformations that take (image, target) and return (image, target)
    """

    _children_names: list[str] = ["sample_transforms"]

    def __init__(self, transforms: list[Callable[[Any, Any], tuple[Any, Any]]]) -> None:
        self.sample_transforms = transforms

    def __call__(self, x: Any, target: Any) -> tuple[Any, Any]:
        for t in self.sample_transforms:
            x, target = t(x, target)
        return x, target


class ImageTransform(NestedObject):
    """
    Wraps a transformation that applies only to images to also return the unchanged target.

    Args:
        transform: Callable that transforms an image.
    """

    _children_names: list[str] = ["img_transform"]

    def __init__(self, transform: Callable[[Any], Any]) -> None:
        self.img_transform = transform

    def __call__(self, img: Any, target: Any) -> tuple[Any, Any]:
        img = self.img_transform(img)
        return img, target


class ColorInversion(NestedObject):
    """
    Converts the image to grayscale, colorizes it with random low RGB values, and inverts colors.

    Args:
        min_val: Minimum value for RGB colorization (between 0 and 1).
    """

    def __init__(self, min_val: float = 0.5) -> None:
        self.min_val = min_val

    def extra_repr(self) -> str:
        return f"min_val={self.min_val}"

    def __call__(self, img: Any) -> Any:
        return F.invert_colors(img, self.min_val)


class OneOf(NestedObject):
    """
    Randomly applies one transformation from a list of options.

    Args:
        transforms: List of transformations, one will be chosen at random during each call.
    """

    _children_names: list[str] = ["transforms"]

    def __init__(self, transforms: list[Callable[[Any], Any]]) -> None:
        self.transforms = transforms

    def __call__(self, img: Any, target: np.ndarray | None = None) -> Any | tuple[Any, np.ndarray]:
        transfo = random.choice(self.transforms)
        return transfo(img) if target is None else transfo(img, target)  # type: ignore[call-arg]


class RandomApply(NestedObject):
    """
    Applies a transformation with a given probability.

    Args:
        transform: Transformation function to conditionally apply.
        p: Probability of applying the transform (between 0 and 1).
    """

    def __init__(self, transform: Callable[[Any], Any], p: float = 0.5) -> None:
        self.transform = transform
        self.p = p

    def extra_repr(self) -> str:
        return f"transform={self.transform}, p={self.p}"

    def __call__(self, img: Any, target: np.ndarray | None = None) -> Any | tuple[Any, np.ndarray]:
        if random.random() < self.p:
            return self.transform(img) if target is None else self.transform(img, target)  # type: ignore[call-arg]
        return img if target is None else (img, target)


class RandomRotate(NestedObject):
    """
    Rotates the image and its bounding boxes by a random angle within a specified range.

    Args:
        max_angle: Max angle (in degrees) for rotation. Rotation sampled from [-max_angle, max_angle].
        expand: Whether to expand the image canvas to fit the entire rotated image.
    """

    def __init__(self, max_angle: float = 5.0, expand: bool = False) -> None:
        self.max_angle = max_angle
        self.expand = expand

    def extra_repr(self) -> str:
        return f"max_angle={self.max_angle}, expand={self.expand}"

    def __call__(self, img: Any, target: np.ndarray) -> tuple[Any, np.ndarray]:
        angle = random.uniform(-self.max_angle, self.max_angle)
        r_img, r_polys = F.rotate_sample(img, target, angle, self.expand)
        # Keep only boxes that are valid (non-degenerate)
        is_kept = (r_polys.max(1) > r_polys.min(1)).sum(1) == 2
        return r_img, r_polys[is_kept]


class RandomCrop(NestedObject):
    """
    Randomly crops the image and adjusts the bounding boxes accordingly.

    Args:
        scale: Tuple specifying min and max area of the crop (relative to original image).
        ratio: Tuple specifying min and max aspect ratio (height/width) of the crop.
    """

    def __init__(self, scale: tuple[float, float] = (0.08, 1.0), ratio: tuple[float, float] = (0.75, 1.33)) -> None:
        self.scale = scale
        self.ratio = ratio

    def extra_repr(self) -> str:
        return f"scale={self.scale}, ratio={self.ratio}"

    def __call__(self, img: Any, target: np.ndarray) -> tuple[Any, np.ndarray]:
        scale = random.uniform(*self.scale)
        ratio = random.uniform(*self.ratio)

        height, width = img.shape[:2]
        crop_area = scale * height * width
        aspect_ratio = ratio * (width / height)

        crop_width = int(round(math.sqrt(crop_area * aspect_ratio)))
        crop_height = int(round(math.sqrt(crop_area / aspect_ratio)))

        crop_width = min(crop_width, width)
        crop_height = min(crop_height, height)

        x = random.randint(0, width - crop_width)
        y = random.randint(0, height - crop_height)

        # Normalized crop box coordinates: (x_min, y_min, x_max, y_max)
        crop_box = (x / width, y / height, (x + crop_width) / width, (y + crop_height) / height)

        # Convert polygon boxes (4 points) to bounding boxes if needed
        if target.shape[1:] == (4, 2):
            min_xy = np.min(target, axis=1)
            max_xy = np.max(target, axis=1)
            _target = np.concatenate((min_xy, max_xy), axis=1)
        else:
            _target = target

        cropped_img, crop_boxes = F.crop_detection(img, _target, crop_box)

        # If no valid box remains, return original
        if crop_boxes.shape[0] == 0:
            return img, target

        return cropped_img, np.clip(crop_boxes, 0, 1)
