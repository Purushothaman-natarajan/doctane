import os
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Callable
from PIL import Image
import torchvision.transforms.functional as TF

# Internal dependencies
from doctane.utils.io_elements import Document
from doctane.ocr_pipeline.utils import get_language
from doctane.models.detection.detection_predictor import DetectionPredictor
from doctane.models.recognition.recognition_predictor import RecognitionPredictor
from doctane.utils.builder import DocumentBuilder
from doctane.utils.geometry import (
    detach_scores, extract_crops, extract_rcrops,
    remove_image_padding, rotate_image
)
from doctane.ocr_pred_pipe.utils import estimate_orientation, rectify_crops, rectify_loc_preds


__all__ = ["OCRPredictor"]


def pad_to_divisible(tensor: torch.Tensor, divisor: int = 32) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pads image tensor so height and width are divisible by a given number."""
    h, w = tensor.shape[-2:]
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
    return padded, (h, w)


class OCRPredictor(nn.Module):
    """
    Unified OCR predictor for document text detection and recognition.
    """

    def __init__(
        self,
        det_predictor: DetectionPredictor,
        reco_predictor: RecognitionPredictor,
        assume_straight_pages: bool = True,
        straighten_pages: bool = False,
        preserve_aspect_ratio: bool = True,
        symmetric_pad: bool = True,
        detect_orientation: bool = False,
        detect_language: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.det_predictor = det_predictor.eval()
        self.reco_predictor = reco_predictor.eval()

        self.assume_straight_pages = assume_straight_pages
        self.straighten_pages = straighten_pages
        self.detect_orientation = detect_orientation
        self.detect_language = detect_language
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad
        self.hooks: list[Callable] = []

        # Load optional orientation predictors if needed
        self._page_orientation_disabled = kwargs.pop("disable_page_orientation", False)
        self._crop_orientation_disabled = kwargs.pop("disable_crop_orientation", False)

        self.page_orientation_predictor = (
            page_orientation_predictor(pretrained=True, disabled=self._page_orientation_disabled)
            if detect_orientation or straighten_pages or not assume_straight_pages else None
        )

        self.crop_orientation_predictor = (
            None if assume_straight_pages else crop_orientation_predictor(pretrained=True, disabled=self._crop_orientation_disabled)
        )

        self.doc_builder = DocumentBuilder(**kwargs)

    def add_hook(self, hook: Callable) -> None:
        """Register a post-processing hook."""
        self.hooks.append(hook)

    def _general_page_orientations(self, pages: list[np.ndarray]) -> list[tuple[int, float]]:
        """Predicts the general orientation of each page."""
        _, classes, probs = zip(self.page_orientation_predictor(pages))  # type: ignore[misc]
        return [(orientation, prob) for cls, prob in zip(classes, probs) for orientation, prob in zip(cls, prob)]

    def _get_orientations(
        self, pages: list[np.ndarray], seg_maps: list[np.ndarray]
    ) -> tuple[list[tuple[int, float]], list[int]]:
        """Returns predicted orientations for pages and crops."""
        general = self._general_page_orientations(pages)
        origin = [estimate_orientation(s, g) for s, g in zip(seg_maps, general)]
        return general, origin

    def _straighten_pages(
        self,
        pages: list[np.ndarray],
        seg_maps: list[np.ndarray],
        general: list[tuple[int, float]] | None = None,
        origin: list[int] | None = None,
    ) -> list[np.ndarray]:
        """Rotates pages to correct general orientation."""
        general = general or self._general_page_orientations(pages)
        origin = origin or [estimate_orientation(s, g) for s, g in zip(seg_maps, general)]
        return [remove_image_padding(rotate_image(p, angle, expand=True)) for p, angle in zip(pages, origin)]

    @staticmethod
    def _generate_crops(
        pages: list[np.ndarray],
        loc_preds: list[np.ndarray],
        channels_last: bool,
        assume_straight_pages: bool = False,
        assume_horizontal: bool = False,
    ) -> list[list[np.ndarray]]:
        """Extracts crops based on box predictions."""
        if assume_straight_pages:
            return [extract_crops(p, b[:, :4], channels_last) for p, b in zip(pages, loc_preds)]
        return [extract_rcrops(p, b[:, :4], channels_last, assume_horizontal) for p, b in zip(pages, loc_preds)]

    @staticmethod
    def _prepare_crops(
        pages: list[np.ndarray],
        loc_preds: list[np.ndarray],
        channels_last: bool,
        assume_straight_pages: bool = False,
        assume_horizontal: bool = False,
    ) -> tuple[list[list[np.ndarray]], list[np.ndarray]]:
        """Prepares valid crops and updates their boxes accordingly."""
        crops = OCRPredictor._generate_crops(pages, loc_preds, channels_last, assume_straight_pages, assume_horizontal)
        is_kept = [[crop.shape[0] > 0 and crop.shape[1] > 0 for crop in pc] for pc in crops]
        crops = [[c for c, keep in zip(pc, pk) if keep] for pc, pk in zip(crops, is_kept)]
        loc_preds = [boxes[kept] for boxes, kept in zip(loc_preds, is_kept)]
        return crops, loc_preds

    def _rectify_crops(
        self,
        crops: list[list[np.ndarray]],
        loc_preds: list[np.ndarray],
    ) -> tuple[list[list[np.ndarray]], list[np.ndarray], list[tuple[int, float]]]:
        """Rotates crops using crop orientation predictor."""
        orientations, classes, probs = zip(*[self.crop_orientation_predictor(pc) for pc in crops])  # type: ignore[misc]
        rectified_crops = [rectify_crops(pc, o) for pc, o in zip(crops, orientations)]
        rectified_preds = [rectify_loc_preds(lp, o) for lp, o in zip(loc_preds, orientations)]
        crop_orientations = [(o, p) for c, p in zip(classes, probs) for o, p in zip(c, p)]
        return rectified_crops, rectified_preds, crop_orientations

    @staticmethod
    def _process_predictions(
        loc_preds: list[np.ndarray],
        word_preds: list[tuple[str, float]],
        crop_orientations: list[dict[str, Any]]
    ) -> tuple[list[np.ndarray], list[list[tuple[str, float]]], list[list[dict[str, Any]]]]:
        """Organizes predictions by page."""
        text_preds, crop_orient_preds = [], []
        idx = 0
        for boxes in loc_preds:
            n = boxes.shape[0]
            text_preds.append(word_preds[idx:idx + n])
            crop_orient_preds.append(crop_orientations[idx:idx + n])
            idx += n
        return loc_preds, text_preds, crop_orient_preds

    @torch.inference_mode()
    def forward(
        self,
        pages: list[np.ndarray | torch.Tensor],
        debug_dir: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Main forward pass for OCR pipeline."""
        if any(p.ndim != 3 for p in pages):
            raise ValueError("All input pages must be 3D tensors or arrays.")

        # Normalize & stack input pages
        orig_shapes = [p.shape[:2] if isinstance(p, np.ndarray) else p.shape[-2:] for p in pages]
        if isinstance(pages[0], np.ndarray):
            pages_tensor = torch.from_numpy(np.stack(pages)).permute(0, 3, 1, 2).float() / 255.0
        else:
            pages_tensor = torch.stack(pages)

        padded_pages = []
        for img in pages_tensor:
            padded_img, _ = pad_to_divisible(img)
            padded_pages.append(padded_img)
        pages_tensor = torch.stack(padded_pages)

        # Run text detection
        out_dict = self.det_predictor(pages_tensor, return_model_output=True, return_preds=True)
        loc_preds = out_dict["preds"]
        out_maps = out_dict["out_map"]

        seg_maps = [
            np.where(out_map > getattr(self.det_predictor.postprocessor, "bin_thresh"), 255, 0).astype(np.uint8)
            for out_map in out_maps
        ]

        # Orientation correction
        orientations = None
        if self.detect_orientation:
            general_orient, origin_orient = self._get_orientations(pages, seg_maps)
            orientations = [{"value": o, "confidence": None} for o in origin_orient]

        if self.straighten_pages:
            pages = self._straighten_pages(pages, seg_maps, general_orient, origin_orient)

        loc_preds = [list(pred.values())[0] for pred in loc_preds]
        loc_preds, objectness_scores = detach_scores(loc_preds)

        for hook in self.hooks:
            loc_preds = hook(loc_preds)

        channels_last = isinstance(pages[0], np.ndarray)
        crops, loc_preds = self._prepare_crops(
            pages,
            loc_preds,
            channels_last,
            assume_straight_pages=self.assume_straight_pages,
            assume_horizontal=self._page_orientation_disabled,
        )

        # Debug: save crops for debugging :: 
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            for i, page_crops in enumerate(crops):
                for j, crop in enumerate(page_crops):
                    if isinstance(crop, torch.Tensor):
                        crop = crop.permute(1, 2, 0).detach().cpu().numpy()
                    crop = (crop * 255).astype(np.uint8) if crop.max() <= 1.0 else crop
                    out_path = os.path.join(debug_dir, f"page{i:02d}_crop{j:04d}.png")
                    cv2.imwrite(out_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        crop_orientations = []
        if not self.assume_straight_pages:
            crops, loc_preds, _crop_orientations = self._rectify_crops(crops, loc_preds)
            crop_orientations = [{"value": o, "confidence": p} for o, p in _crop_orientations]

        all_crops = [crop for page_crops in crops for crop in page_crops]
        word_preds = RecognitionPredictor(self.reco_predictor)(all_crops)

        if not crop_orientations:
            crop_orientations = [{"value": 0, "confidence": None} for _ in word_preds]

        boxes, text_preds, crop_orientations = self._process_predictions(loc_preds, word_preds, crop_orientations)

        languages_dict = None
        if self.detect_language:
            langs = [get_language(" ".join([t[0] for t in pred])) for pred in text_preds]
            languages_dict = [{"value": lang[0], "confidence": lang[1]} for lang in langs]

        return self.doc_builder(
            pages,
            boxes,
            objectness_scores,
            text_preds,
            orig_shapes,
            crop_orientations,
            orientations,
            languages_dict,
        )
