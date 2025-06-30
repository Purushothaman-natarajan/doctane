import math
import numpy as np
import torch
from PIL.Image import Image
from scipy.ndimage import gaussian_filter
from torch.nn.functional import pad
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T

from doctane.utils.transforms.functional.pytorch import random_shadow

__all__ = [
    "Resize",
    "GaussianNoise",
    "ChannelShuffle",
    "RandomHorizontalFlip",
    "RandomShadow",
    "RandomResize",
    "GaussianBlur",
]


class Resize(T.Resize):
    """
    Resize the input image to the given size, with options to preserve aspect ratio and pad symmetrically.

    Args:
        size (int or tuple): Desired output size.
        interpolation: Interpolation method.
        preserve_aspect_ratio (bool): Whether to preserve original aspect ratio.
        symmetric_pad (bool): Whether to symmetrically pad to maintain target shape.
    """
    def __init__(
        self,
        size: int | tuple[int, int],
        interpolation=F.InterpolationMode.BILINEAR,
        preserve_aspect_ratio: bool = False,
        symmetric_pad: bool = False,
    ) -> None:
        super().__init__(size, interpolation, antialias=True)
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad

        if not isinstance(self.size, (int, tuple, list)):
            raise AssertionError("size should be either a tuple, a list or an int")

    def forward(
        self,
        img: torch.Tensor,
        target: np.ndarray | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, np.ndarray]:
        if isinstance(self.size, int):
            target_ratio = img.shape[-2] / img.shape[-1]
        else:
            target_ratio = self.size[0] / self.size[1]
        actual_ratio = img.shape[-2] / img.shape[-1]

        if not self.preserve_aspect_ratio or (target_ratio == actual_ratio and isinstance(self.size, (tuple, list))):
            return (super().forward(img), target) if target is not None else super().forward(img)

        # Compute new size preserving aspect ratio
        if isinstance(self.size, (tuple, list)):
            tmp_size = (self.size[0], max(int(self.size[0] / actual_ratio), 1)) if actual_ratio > target_ratio else (max(int(self.size[1] * actual_ratio), 1), self.size[1])
        else:
            tmp_size = (max(int(self.size * actual_ratio), 1), self.size) if img.shape[-2] <= img.shape[-1] else (self.size, max(int(self.size / actual_ratio), 1))

        img = F.resize(img, tmp_size, self.interpolation, antialias=True)
        raw_shape = img.shape[-2:]

        if isinstance(self.size, (tuple, list)):
            _pad = (0, self.size[1] - img.shape[-1], 0, self.size[0] - img.shape[-2])
            if self.symmetric_pad:
                half_pad = (math.ceil(_pad[1] / 2), math.ceil(_pad[3] / 2))
                _pad = (half_pad[0], _pad[1] - half_pad[0], half_pad[1], _pad[3] - half_pad[1])
            img = pad(img, _pad)

        if target is not None:
            if self.symmetric_pad:
                offset = half_pad[0] / img.shape[-1], half_pad[1] / img.shape[-2]

            if self.preserve_aspect_ratio:
                if target.shape[1:] == (4,):
                    if isinstance(self.size, (tuple, list)) and self.symmetric_pad:
                        target[:, [0, 2]] = offset[0] + target[:, [0, 2]] * raw_shape[-1] / img.shape[-1]
                        target[:, [1, 3]] = offset[1] + target[:, [1, 3]] * raw_shape[-2] / img.shape[-2]
                    else:
                        target[:, [0, 2]] *= raw_shape[-1] / img.shape[-1]
                        target[:, [1, 3]] *= raw_shape[-2] / img.shape[-2]
                elif target.shape[1:] == (4, 2):
                    if isinstance(self.size, (tuple, list)) and self.symmetric_pad:
                        target[..., 0] = offset[0] + target[..., 0] * raw_shape[-1] / img.shape[-1]
                        target[..., 1] = offset[1] + target[..., 1] * raw_shape[-2] / img.shape[-2]
                    else:
                        target[..., 0] *= raw_shape[-1] / img.shape[-1]
                        target[..., 1] *= raw_shape[-2] / img.shape[-2]
                else:
                    raise AssertionError("Boxes should be in the format (n_boxes, 4, 2) or (n_boxes, 4)")

            return img, np.clip(target, 0, 1)

        return img

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        _repr = f"output_size={self.size}, interpolation='{interpolate_str}'"
        if self.preserve_aspect_ratio:
            _repr += f", preserve_aspect_ratio={self.preserve_aspect_ratio}, symmetric_pad={self.symmetric_pad}"
        return f"{self.__class__.__name__}({_repr})"


class GaussianNoise(torch.nn.Module):
    """
    Adds Gaussian noise to the input tensor.

    Args:
        mean (float): Mean of the Gaussian distribution.
        std (float): Standard deviation.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = self.mean + 2 * self.std * torch.rand(x.shape, device=x.device) - self.std
        if x.dtype == torch.uint8:
            return (x + 255 * noise).round().clamp(0, 255).to(dtype=torch.uint8)
        return (x + noise.to(dtype=x.dtype)).clamp(0, 1)

    def extra_repr(self) -> str:
        return f"mean={self.mean}, std={self.std}"


class GaussianBlur(torch.nn.Module):
    """
    Applies Gaussian blur with a random sigma from a given range.

    Args:
        sigma (tuple): Range of standard deviation values.
    """

    def __init__(self, sigma: tuple[float, float]) -> None:
        super().__init__()
        self.sigma_range = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = torch.empty(1).uniform_(*self.sigma_range).item()
        return torch.tensor(
            gaussian_filter(x.numpy(), sigma=sigma, mode="reflect", truncate=4.0),
            dtype=x.dtype,
            device=x.device,
        )


class ChannelShuffle(torch.nn.Module):
    """Randomly shuffles the channels of an image."""

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        chan_order = torch.rand(img.shape[0]).argsort()
        return img[chan_order]


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    """Randomly flips the image horizontally and adjusts target bounding boxes accordingly."""

    def forward(self, img: torch.Tensor | Image, target: np.ndarray) -> tuple[torch.Tensor | Image, np.ndarray]:
        if torch.rand(1) < self.p:
            _img = F.hflip(img)
            _target = target.copy()
            if target.shape[1:] == (4,):
                _target[:, ::2] = 1 - target[:, [2, 0]]
            else:
                _target[..., 0] = 1 - target[..., 0]
            return _img, _target
        return img, target


class RandomShadow(torch.nn.Module):
    """
    Applies random shadowing to an image.

    Args:
        opacity_range (tuple): Range for shadow opacity.
    """

    def __init__(self, opacity_range: tuple[float, float] | None = None) -> None:
        super().__init__()
        self.opacity_range = opacity_range if opacity_range else (0.2, 0.8)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        try:
            if x.dtype == torch.uint8:
                return (255 * random_shadow(x.float() / 255, self.opacity_range)).round().clip(0, 255).to(torch.uint8)
            return random_shadow(x, self.opacity_range).clip(0, 1)
        except ValueError:
            return x

    def extra_repr(self) -> str:
        return f"opacity_range={self.opacity_range}"


class RandomResize(torch.nn.Module):
    """
    Randomly resizes the image and adjusts the corresponding targets.

    Args:
        scale_range (tuple): Range of scale for height and width.
        preserve_aspect_ratio (bool or float): If float, treated as probability.
        symmetric_pad (bool or float): If float, treated as probability.
        p (float): Probability of applying transformation.
    """

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.3, 0.9),
        preserve_aspect_ratio: bool | float = False,
        symmetric_pad: bool | float = False,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        self.scale_range = scale_range
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad
        self.p = p
        self._resize = Resize

    def forward(self, img: torch.Tensor, target: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        if torch.rand(1) < self.p:
            scale_h = np.random.uniform(*self.scale_range)
            scale_w = np.random.uniform(*self.scale_range)
            new_size = (int(img.shape[-2] * scale_h), int(img.shape[-1] * scale_w))

            _img, _target = self._resize(
                new_size,
                preserve_aspect_ratio=self.preserve_aspect_ratio
                if isinstance(self.preserve_aspect_ratio, bool)
                else bool(torch.rand(1) <= self.preserve_aspect_ratio),
                symmetric_pad=self.symmetric_pad
                if isinstance(self.symmetric_pad, bool)
                else bool(torch.rand(1) <= self.symmetric_pad),
            )(img, target)

            return _img, _target
        return img, target

    def extra_repr(self) -> str:
        return (
            f"scale_range={self.scale_range}, "
            f"preserve_aspect_ratio={self.preserve_aspect_ratio}, "
            f"symmetric_pad={self.symmetric_pad}, p={self.p}"
        )
