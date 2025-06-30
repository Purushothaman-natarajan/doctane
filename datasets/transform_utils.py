import numpy as np
from PIL import Image
import torch

def convert_to_relative_coords(geoms: np.ndarray, img_shape: tuple[int, int]) -> np.ndarray:
    """Convert a geometry to relative coordinates

    Args:
        geoms: a set of polygons of shape (N, 4, 2) or of straight boxes of shape (N, 4)
        img_shape: the height and width of the image

    Returns:
        the updated geometry
    """
    # Polygon
    if geoms.ndim == 3 and geoms.shape[1:] == (4, 2):
        polygons: np.ndarray = np.empty(geoms.shape, dtype=np.float32)
        polygons[..., 0] = geoms[..., 0] / img_shape[1]
        polygons[..., 1] = geoms[..., 1] / img_shape[0]
        return polygons.clip(0, 1)
    if geoms.ndim == 2 and geoms.shape[1] == 4:
        boxes: np.ndarray = np.empty(geoms.shape, dtype=np.float32)
        boxes[:, ::2] = geoms[:, ::2] / img_shape[1]
        boxes[:, 1::2] = geoms[:, 1::2] / img_shape[0]
        return boxes.clip(0, 1)

    raise ValueError(f"invalid format for arg `geoms`: {geoms.shape}")

def get_img_shape(img: torch.Tensor) -> tuple[int, int]:
    """Get the shape (H, W) of an image tensor."""
    return img.shape[-2:]

def pre_transform_multiclass(img, target: tuple[np.ndarray, list]) -> tuple[np.ndarray, dict[str, list]]:
    """Converts multiclass target to relative coordinates.

    Args:
        img: Image
        target: tuple of target polygons and their classes names

    Returns:
        Image and dictionary of boxes, with class names as keys
    """
    boxes = convert_to_relative_coords(target[0], get_img_shape(img))
    boxes_classes = target[1]
    boxes_dict: dict = {k: [] for k in sorted(set(boxes_classes))}
    for k, poly in zip(boxes_classes, boxes):
        boxes_dict[k].append(poly)
    boxes_dict = {k: np.stack(v, axis=0) for k, v in boxes_dict.items()}
    return img, boxes_dict