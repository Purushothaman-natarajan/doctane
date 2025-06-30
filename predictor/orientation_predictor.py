import numpy as np
import torch
from torch import nn
from typing import Optional, Union

from doctane.models.pre_processor import PreProcessor
from doctane.utils.dl_utils import set_device_and_dtype

__all__ = ["OrientationPredictor"]

class OrientationPredictor(nn.Module):
    """
    Detects the reading orientation of a text region or page.
    Supports 4 possible angles: 0, 90, 180, 270 degrees (counter-clockwise).

    Args:
        pre_processor (PreProcessor | None): Transforms raw inputs into model-ready tensors.
        model (nn.Module | None): A classification model with a cfg dict containing 'classes'.
    """

    def __init__(
        self,
        pre_processor: Optional[PreProcessor],
        model: Optional[nn.Module],
    ) -> None:
        super().__init__()
        self.pre_processor = pre_processor if isinstance(pre_processor, PreProcessor) else None
        self.model = model.eval() if isinstance(model, nn.Module) else None

    @torch.inference_mode()
    def forward(
        self,
        inputs: list[Union[np.ndarray, torch.Tensor]],
    ) -> list[list[Union[int, float]]]:
        # Validate input dimensionality: should be (C, H, W)
        if any(input.ndim != 3 for input in inputs):
            raise ValueError("All inputs must be 3D tensors representing multi-channel 2D images.")

        # Return default values if model is disabled
        if self.model is None or self.pre_processor is None:
            n = len(inputs)
            return [[0] * n, [0] * n, [1.0] * n]

        # Pre-process input images
        processed_batches = self.pre_processor(inputs)

        # Ensure model and data are on the same device and dtype
        _params = next(self.model.parameters())
        self.model, processed_batches = set_device_and_dtype(
            self.model, processed_batches, _params.device, _params.dtype
        )

        # Run inference
        logits_batches = [self.model(batch) for batch in processed_batches]  # type: ignore

        # Get max confidence probabilities
        confidences = [
            torch.max(torch.softmax(logits, dim=1), dim=1).values.cpu().numpy()
            for logits in logits_batches
        ]

        # Get predicted class indices
        predicted_classes = [
            logits.argmax(dim=1).cpu().numpy()
            for logits in logits_batches
        ]

        # Flatten predictions
        class_idxs = [int(idx) for batch in predicted_classes for idx in batch]
        confs = [round(float(p), 2) for batch in confidences for p in batch]

        # Resolve class labels using model configuration
        class_map = getattr(self.model, "cfg", {}).get("classes", {})
        classes = [int(class_map.get(idx, 0)) for idx in class_idxs]

        return [class_idxs, classes, confs]
