from collections.abc import Sequence
from typing import Any, List, Tuple, Union

import numpy as np
import torch
from torch import nn


from doctane.models.recognition.pre_processor import PreProcessor
from doctane.utils.dl_utils import set_device_and_dtype
from doctane.models.recognition.string_utils import merge_multi_strings


__all__ = ["RecognitionPredictor", "split_crops", "remap_preds"]


def split_crops(
    crops: List[np.ndarray],
    max_ratio: float,
    target_ratio: int,
    dilation: float,
    channels_last: bool = True,
) -> Tuple[List[np.ndarray], List[Union[int, Tuple[int, int]]], bool]:
    """Chunk crops horizontally to match a given aspect ratio.

    Args:
        crops: List of numpy arrays representing image crops, with shape (H, W, 3) if channels_last or (3, H, W) otherwise.
        max_ratio: The maximum aspect ratio that won't trigger the chunking.
        target_ratio: The aspect ratio to which crops should be chunked.
        dilation: The width dilation factor for final chunks (provides overlap between crops).
        channels_last: Whether the numpy array has the channels in the last dimension.

    Returns:
        A tuple containing:
            - new crops after chunking (if applicable),
            - crop map indicating the relationships between original and new crops,
            - a boolean indicating whether any remapping is required.
    """
    _remap_required = False
    crop_map: List[Union[int, Tuple[int, int]]] = []
    new_crops: List[np.ndarray] = []

    for crop in crops:
        h, w = crop.shape[:2] if channels_last else crop.shape[-2:]
        aspect_ratio = w / h

        if aspect_ratio > max_ratio:
            # Determine how many subcrops we need based on the aspect ratio
            num_subcrops = int(aspect_ratio // target_ratio)
            width = dilation * w / num_subcrops
            centers = [(w / num_subcrops) * (0.5 + idx) for idx in range(num_subcrops)]

            # Generate subcrops
            if channels_last:
                _crops = [
                    crop[:, max(0, int(round(center - width / 2))) : min(w - 1, int(round(center + width / 2))), :]
                    for center in centers
                ]
            else:
                _crops = [
                    crop[:, :, max(0, int(round(center - width / 2))) : min(w - 1, int(round(center + width / 2)))]
                    for center in centers
                ]

            # Remove any zero-sized crops
            _crops = [crop for crop in _crops if all(s > 0 for s in crop.shape)]

            # Record the crop mapping
            crop_map.append((len(new_crops), len(new_crops) + len(_crops)))
            new_crops.extend(_crops)

            # Mark remapping as required
            _remap_required = True
        else:
            crop_map.append(len(new_crops))
            new_crops.append(crop)

    return new_crops, crop_map, _remap_required


def remap_preds(
    preds: List[Tuple[str, float]],
    crop_map: List[Union[int, Tuple[int, int]]],
    dilation: float,
) -> List[Tuple[str, float]]:
    """Remap predictions based on crop splitting.

    Args:
        preds: List of predictions with associated probabilities.
        crop_map: The map indicating the relationship between original and split crops.
        dilation: The dilation factor used in splitting the crops.

    Returns:
        A list of remapped predictions, merging text and minimizing probabilities.
    """
    remapped_out = []
    for _idx in crop_map:
        # If crop wasn't split
        if isinstance(_idx, int):
            remapped_out.append(preds[_idx])
        else:
            # Merge predictions from multiple subcrops
            vals, probs = zip(*preds[_idx[0] : _idx[1]])
            merged_text = merge_multi_strings(vals, dilation)
            min_prob = min(probs)  # Minimize the probability
            remapped_out.append((merged_text, min_prob))

    return remapped_out


class RecognitionPredictor(nn.Module):
    """Model for identifying character sequences in images using a detection architecture.

    Args:
        pre_processor: A preprocessing module for transforming input images before feeding to the model.
        model: The core OCR model.
        split_wide_crops: Whether to enable split processing for wide crops.
        
        - Splits wide crops automatically
        - Batches inputs efficiently
        - Merges predictions from subcrops intelligently
    """
    def __init__(
        self,
        model: nn.Module,
        split_wide_crops: bool = True,
        critical_ar: float = 8.0,
        target_ar: float = 6.0,
        dil_factor: float = 1.4,
        output_size: Tuple[int, int] = (32, 128),
        default_batch_size: int = 4,
    ) -> None:
        super().__init__()
        self.model = model.eval()
        self.split_wide_crops = split_wide_crops
        self.critical_ar = critical_ar
        self.target_ar = target_ar
        self.dil_factor = dil_factor
        self.output_size = output_size
        self.default_batch_size = default_batch_size

        # Create a placeholder pre_processor; batch_size will be set per forward call
        self._base_pre_processor = lambda batch_size: PreProcessor(
            output_size=self.output_size,
            batch_size=batch_size,
            preserve_aspect_ratio=True,
            symmetric_pad=True,
        )

    @torch.inference_mode()
    def forward(
        self,
        crops: Sequence[Union[np.ndarray, torch.Tensor]],
        use_avg_conf: bool = False,
        **kwargs: Any,
    ) -> List[Tuple[str, float]]:
        if not crops:
            return []

        for i, crop in enumerate(crops):
            if not isinstance(crop, (np.ndarray, torch.Tensor)) or crop.ndim != 3:
                raise ValueError(f"Crop {i} has invalid format. Expected 3D image.")

        remapped = False
        if self.split_wide_crops:
            crops, crop_map, remapped = split_crops(
                crops,
                max_ratio=self.critical_ar,
                target_ratio=self.target_ar,
                dilation=self.dil_factor,
                channels_last=isinstance(crops[0], np.ndarray),
            )

        # Set batch size dynamically
        dynamic_batch_size = min(len(crops), self.default_batch_size)
        pre_processor = self._base_pre_processor(dynamic_batch_size)

        # Preprocess and batch crops
        batch_tensor = pre_processor(crops)

        # Device alignment
        _params = next(self.model.parameters())
        self.model, batch_tensor = set_device_and_dtype(
            self.model, batch_tensor, _params.device, _params.dtype
        )

        # Run inference
        raw = [self.model(batch, return_preds=True, **kwargs)["preds"] for batch in batch_tensor]
        out = [charseq for batch in raw for charseq in batch]

        # Optionally remap
        if self.split_wide_crops and remapped:
            out = self._remap_preds(out, crop_map, self.dil_factor, use_avg_conf)

        return out

    def _remap_preds(
        self,
        preds: List[Tuple[str, float]],
        crop_map: List[Union[int, Tuple[int, int]]],
        dilation: float,
        use_avg_conf: bool,
    ) -> List[Tuple[str, float]]:
        remapped_out = []
        for idx in crop_map:
            if isinstance(idx, int):
                remapped_out.append(preds[idx])
            else:
                sub_preds = preds[idx[0]: idx[1]]
                texts, confs = zip(*sub_preds)
                merged_text = merge_multi_strings(texts, dilation)
                merged_conf = (
                    sum(confs) / len(confs) if use_avg_conf else min(confs)
                )
                remapped_out.append((merged_text, merged_conf))
        return remapped_out
