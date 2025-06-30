import cv2
import numpy as np
import pyclipper

from doctane.utils.representation import NestedObject


__all__ = ["DetectionPostProcessor"]


class DetectionPostProcessor(NestedObject):
    def __init__(self, box_thresh: float = 0.1, bin_thresh: float = 0.1, assume_straight_pages: bool = True) -> None:
        self.box_thresh = box_thresh
        self.bin_thresh = bin_thresh
        self.assume_straight_pages = assume_straight_pages
        self._opening_kernel: np.ndarray = np.ones((3, 3), dtype=np.uint8)        
        self.unclip_ratio = 1.5

    def extra_repr(self) -> str:
        return f"bin_thresh={self.bin_thresh}, box_thresh={self.box_thresh}"

    @staticmethod
    def box_score(pred: np.ndarray, points: np.ndarray, assume_straight_pages: bool = True) -> float:
        h, w = pred.shape[:2]
        if assume_straight_pages:
            xmin = np.clip(np.floor(points[:, 0].min()).astype(np.int32), 0, w - 1)
            xmax = np.clip(np.ceil(points[:, 0].max()).astype(np.int32), 0, w - 1)
            ymin = np.clip(np.floor(points[:, 1].min()).astype(np.int32), 0, h - 1)
            ymax = np.clip(np.ceil(points[:, 1].max()).astype(np.int32), 0, h - 1)
            return pred[ymin : ymax + 1, xmin : xmax + 1].mean()
        else:
            mask: np.ndarray = np.zeros((h, w), dtype=np.int32)
            cv2.fillPoly(mask, [points.astype(np.int32)], 1)
            product = pred * mask
            return np.sum(product) / np.count_nonzero(product)

    def __call__(self, proba_map) -> list[list[np.ndarray]]:
        if proba_map.ndim != 4:
            raise AssertionError(f"`proba_map` must be 4-dimensional (N, H, W, C), got {proba_map.ndim}.")
        bin_map = [
            [cv2.morphologyEx(bmap[..., idx], cv2.MORPH_OPEN, self._opening_kernel)
             for idx in range(proba_map.shape[-1])]
            for bmap in (proba_map >= self.bin_thresh).astype(np.uint8)
        ]
        return [
            [self.bitmap_to_boxes(pmaps[..., idx], bmaps[idx]) for idx in range(proba_map.shape[-1])]
            for pmaps, bmaps in zip(proba_map, bin_map)
        ]

    def polygon_to_box(self, points: np.ndarray) -> np.ndarray:
        if not self.assume_straight_pages:
            rect = cv2.minAreaRect(points)
            points = cv2.boxPoints(rect)
            area = (rect[1][0] + 1) * (1 + rect[1][1])
            length = 2 * (rect[1][0] + rect[1][1]) + 2
        else:
            area = cv2.contourArea(points)
            length = cv2.arcLength(points, closed=True)

        distance = area * self.unclip_ratio / length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(points.astype(np.int32), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        _points = offset.Execute(distance)

        if not _points:
            return None

        expanded_points = np.asarray(_points[0])
        return (cv2.boundingRect(expanded_points)
                if self.assume_straight_pages
                else np.roll(cv2.boxPoints(cv2.minAreaRect(expanded_points)), -1, axis=0))

    def bitmap_to_boxes(self, pred: np.ndarray, bitmap: np.ndarray) -> np.ndarray:
        height, width = bitmap.shape[:2]
        boxes = []
        contours, _ = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if np.any(contour[:, 0].max(axis=0) - contour[:, 0].min(axis=0) < 2):
                continue

            if self.assume_straight_pages:
                x, y, w, h = cv2.boundingRect(contour)
                points = np.array([[x, y], [x, y + h], [x + w, y + h], [x + w, y]])
                score = self.box_score(pred, points, True)
            else:
                score = self.box_score(pred, contour, False)

            if score < self.box_thresh:
                continue

            if self.assume_straight_pages:
                _box = self.polygon_to_box(points)
                if _box is None:
                    continue
                x, y, w, h = _box
                xmin, ymin, xmax, ymax = x / width, y / height, (x + w) / width, (y + h) / height
                boxes.append([xmin, ymin, xmax, ymax, score])
            else:
                _box = self.polygon_to_box(np.squeeze(contour))
                if _box is None:
                    continue
                _box[:, 0] /= width
                _box[:, 1] /= height
                boxes.append(np.vstack([_box, np.array([0.0, score])]))

        if not self.assume_straight_pages:
            return np.clip(np.asarray(boxes), 0, 1) if boxes else np.zeros((0, 5, 2), dtype=pred.dtype)
        else:
            return np.clip(np.asarray(boxes), 0, 1) if boxes else np.zeros((0, 5), dtype=pred.dtype)


