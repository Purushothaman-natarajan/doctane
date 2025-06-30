
# Read :: the same script used for text detection can be adapted for row and column
# detection, make sure to change the num_classes to 2 (1 for columns and 1 for rows)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any
import cv2
from shapely.geometry import Polygon
import pyclipper
import segmentation_models_pytorch as smp

from doctane.utils.dl_utils import _bf16_to_float32
from doctane.models.detection.utils import _remove_padding
from doctane.models.detection.detection_postprocessor import DetectionPostProcessor 


class SegmentationModel(nn.Module):
    """
    A segmentation-based detection model wrapper using segmentation_models_pytorch and custom postprocessing.
    """
    def __init__(
        self,
        model_name: str = "Linknet",           # Model architecture (e.g., 'Linknet', 'Unet', etc.)
        encoder_name: str = "resnet50",        # Encoder (backbone) name
        encoder_weights: str = "imagenet",     # Pretrained weights for encoder
        in_channels: int = 3,                  # Input channel count (e.g., 3 for RGB)
        num_classes: int = 2,                  # Number of output classes
        class_names: list[str] | None = None,  # Class labels
        assume_straight_pages: bool = True,    # Flag for postprocessing orientation
        bin_thresh: float = 0.1,               # Binarization threshold
        box_thresh: float = 0.1,               # Box score threshold
        exportable: bool = False,              # Flag for export (e.g., TorchScript, ONNX)
        min_size_box: int = 3,                 # Minimum acceptable bounding box size
        shrink_ratio: float = 0.5              # Polygon shrink ratio for target mask generation
    ):
        super().__init__()

        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.exportable = exportable
        self.assume_straight_pages = assume_straight_pages
        self.min_size_box = min_size_box
        self.shrink_ratio = shrink_ratio

        # Initialize segmentation model from SMP
        self.model = getattr(smp, model_name)(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes,
        )

        # Detection-specific postprocessor
        self.postprocessor = DetectionPostProcessor(
            assume_straight_pages=self.assume_straight_pages,
            bin_thresh=bin_thresh,
            box_thresh=box_thresh,
        )

        # Initialize non-backbone layers
        for n, m in self.named_modules():
            if n.startswith("model.encoder."):
                continue  # Skip pretrained encoder weights
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        target: list[np.ndarray] | None = None,
        return_model_output: bool = False,
        return_preds: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Forward pass with optional output and predictions.

        Args:
            x: input tensor of shape (B, C, H, W)
            target: ground truth (optional)
            return_model_output: flag to return raw model probability maps
            return_preds: flag to return postprocessed detections

        Returns:
            dict containing logits, loss, preds, or out_map depending on flags
        """
        logits = self.model(x)
        
        out: dict[str, Any] = {}
        if self.exportable:
            out["logits"] = logits
            return out

        if return_model_output or target is None or return_preds:
            prob_map = _bf16_to_float32(torch.sigmoid(logits))
        if return_model_output:
            resized_prob_map = F.interpolate(prob_map, size=(1024, 1024), mode="bilinear", align_corners=False)
            out["out_map"] = [t.permute(1, 2, 0).detach().cpu().numpy() for t in resized_prob_map]

        if target is None or return_preds:
            @torch.compiler.disable  # Ensures compatibility with `torch.compile`
            def _postprocess(prob_map: torch.Tensor) -> list[dict[str, Any]]:
                return [
                    dict(zip(self.class_names, preds))
                    for preds in self.postprocessor(
                        prob_map.detach().cpu().permute((0, 2, 3, 1)).numpy()
                    )
                ]
            out["preds"] = _postprocess(prob_map)

        if target is not None:
            loss = self.compute_loss(logits, target)
            out["loss"] = loss

        return out

    def build_target(
        self,
        target: list[dict[str, np.ndarray]],
        output_shape: tuple[int, int, int],
        channels_last: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build ground truth mask and mask validity array from polygon/box annotations.

        Returns:
            seg_target: segmentation ground truth
            seg_mask: mask showing valid areas for loss computation
        """
        if any(t.dtype != np.float32 for tgt in target for t in tgt.values()):
            raise AssertionError("Expected dtype for target entries is 'np.float32'.")
        if any(np.any((t[:, :4] > 1) | (t[:, :4] < 0)) for tgt in target for t in tgt.values()):
            raise ValueError("Target box coordinates should be normalized between 0 and 1.")

        if channels_last:
            h, w, num_classes = output_shape
        else:
            num_classes, h, w = output_shape

        seg_target = np.zeros((len(target), num_classes, h, w), dtype=np.uint8)
        seg_mask = np.ones((len(target), num_classes, h, w), dtype=bool)

        for idx, tgt in enumerate(target):
            for class_idx, _tgt in enumerate(tgt.values()):
                if _tgt.shape[0] == 0:
                    seg_mask[idx, class_idx] = False
                    continue

                abs_boxes = _tgt.copy()

                if abs_boxes.ndim == 3:
                    abs_boxes[:, :, 0] *= w
                    abs_boxes[:, :, 1] *= h
                    polys = abs_boxes
                    boxes_size = np.linalg.norm(abs_boxes[:, 2, :] - abs_boxes[:, 0, :], axis=-1)
                    abs_boxes = np.concatenate((abs_boxes.min(1), abs_boxes.max(1)), -1).round().astype(np.int32)
                else:
                    abs_boxes[:, [0, 2]] *= w
                    abs_boxes[:, [1, 3]] *= h
                    abs_boxes = abs_boxes.round().astype(np.int32)
                    polys = np.stack([
                        abs_boxes[:, [0, 1]],
                        abs_boxes[:, [0, 3]],
                        abs_boxes[:, [2, 3]],
                        abs_boxes[:, [2, 1]],
                    ], axis=1)
                    boxes_size = np.minimum(abs_boxes[:, 2] - abs_boxes[:, 0], abs_boxes[:, 3] - abs_boxes[:, 1])

                for poly, box, box_size in zip(polys, abs_boxes, boxes_size):
                    if box_size < self.min_size_box:
                        seg_mask[idx, class_idx, box[1]:box[3]+1, box[0]:box[2]+1] = False
                        continue

                    polygon = Polygon(poly)
                    distance = polygon.area * (1 - np.power(self.shrink_ratio, 2))  / polygon.length
                    subject = [tuple(p) for p in poly]
                    padding = pyclipper.PyclipperOffset()
                    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                    shrunken = padding.Execute(-distance)

                    if len(shrunken) == 0:
                        seg_mask[idx, class_idx, box[1]:box[3]+1, box[0]:box[2]+1] = False
                        continue

                    shrunken = np.array(shrunken[0]).reshape(-1, 2)
                    if shrunken.shape[0] <= 2 or not Polygon(shrunken).is_valid:
                        seg_mask[idx, class_idx, box[1]:box[3]+1, box[0]:box[2]+1] = False
                        continue

                    cv2.fillPoly(seg_target[idx, class_idx], [shrunken.astype(np.int32)], 1.0)

        if channels_last:
            seg_target = seg_target.transpose((0, 2, 3, 1))
            seg_mask = seg_mask.transpose((0, 2, 3, 1))

        return seg_target, seg_mask

    def compute_loss(
        self,
        out_map: torch.Tensor,
        target: list[np.ndarray],
        gamma: float = 2.0,
        alpha: float = 0.5,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute combined focal + dice loss on segmentation maps.

        Args:
            out_map: model output (logits)
            target: ground truth labels
            gamma: focal loss focusing parameter
            alpha: focal loss class balancing factor
            eps: small epsilon to avoid divide-by-zero

        Returns:
            total loss tensor
        """
        _target, _mask = self.build_target(target, out_map.shape[1:], False)
        seg_target = torch.from_numpy(_target).to(dtype=out_map.dtype, device=out_map.device)
        seg_mask = torch.from_numpy(_mask).to(dtype=torch.float32, device=out_map.device)

        # Binary cross-entropy loss (element-wise)
        bce_loss = F.binary_cross_entropy_with_logits(out_map, seg_target, reduction="none")
        proba_map = torch.sigmoid(out_map)

        if gamma < 0:
            raise ValueError("gamma must be non-negative for focal loss.")

        p_t = proba_map * seg_target + (1 - proba_map) * (1 - seg_target)
        alpha_t = alpha * seg_target + (1 - alpha) * (1 - seg_target)

        focal_loss = alpha_t * (1 - p_t) ** gamma * bce_loss
        focal_loss = (seg_mask * focal_loss).sum((0, 1, 2, 3)) / seg_mask.sum((0, 1, 2, 3))

        # Dice loss computation
        dice_map = torch.softmax(out_map, dim=1) if len(self.class_names) > 1 else proba_map
        inter = (seg_mask * dice_map * seg_target).sum((0, 2, 3))
        cardinality = (seg_mask * (dice_map + seg_target)).sum((0, 2, 3))
        dice_loss = (1 - 2 * inter / (cardinality + eps)).mean()

        return focal_loss + dice_loss
