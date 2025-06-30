import os
from typing import Any

import numpy as np
from doctane.dataset.abstract_dataset import AbstractDataset


__all__ = ["OrientationDataset"]


class OrientationDataset(AbstractDataset):
    """
    Implements a basic image dataset where all targets are filled with zeros (representing 0° rotation).

    Args:
        img_folder (str): Path to the folder containing image files.
        **kwargs: Additional arguments passed to `AbstractDataset`.
    """

    def __init__(self, img_folder: str, **kwargs: Any) -> None:
        if not os.path.isdir(img_folder):
            raise ValueError(f"The specified path is not a directory: {img_folder}")

        super().__init__(img_folder, **kwargs)

        # Supported image file extensions
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        # Filter and prepare dataset
        self.data: list[tuple[str, np.ndarray]] = [
            (img_name, np.array([0], dtype=np.int64))
            for img_name in sorted(os.listdir(self.root))
            if os.path.splitext(img_name)[-1].lower() in valid_extensions
        ]

        if not self.data:
            raise RuntimeError(f"No supported image files found in: {img_folder}")
