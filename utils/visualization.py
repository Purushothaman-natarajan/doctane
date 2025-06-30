import colorsys
from copy import deepcopy
from typing import Any, Union

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.figure import Figure

from doctane.utils.common_types import BoundingBox, Polygon4P

__all__ = ["visualize_page", "visualize_kie_page", "draw_boxes"]

# ------------------------ Patch Creation Utilities ------------------------ #

def rect_patch(
    geometry: BoundingBox,
    page_dimensions: tuple[int, int],
    label: str | None = None,
    color: tuple[float, float, float] = (0, 0, 0),
    alpha: float = 0.3,
    linewidth: int = 2,
    fill: bool = True,
    preserve_aspect_ratio: bool = False,
) -> patches.Rectangle:
    """Create a rectangular patch from a 2-point bounding box."""
    if len(geometry) != 2 or any(len(pt) != 2 for pt in geometry):
        raise ValueError("Geometry must contain two (x, y) points.")

    height, width = page_dimensions
    if preserve_aspect_ratio:
        width = height = max(height, width)

    (xmin, ymin), (xmax, ymax) = geometry
    xmin *= width
    ymin *= height
    w = (xmax - xmin / width) * width
    h = (ymax - ymin / height) * height

    return patches.Rectangle(
        (xmin, ymin),
        w,
        h,
        linewidth=linewidth,
        edgecolor=(*color, alpha),
        facecolor=(*color, alpha) if fill else "none",
        fill=fill,
        label=label,
    )


def polygon_patch(
    geometry: np.ndarray,
    page_dimensions: tuple[int, int],
    label: str | None = None,
    color: tuple[float, float, float] = (0, 0, 0),
    alpha: float = 0.3,
    linewidth: int = 2,
    fill: bool = True,
    preserve_aspect_ratio: bool = False,
) -> patches.Polygon:
    """Create a polygon patch from 4-point rotated geometry."""
    if geometry.shape != (4, 2):
        raise ValueError("Geometry must be a 4x2 ndarray for polygons.")

    height, width = page_dimensions
    scale = max(height, width) if preserve_aspect_ratio else (width, height)
    geometry[:, 0] *= scale[0]
    geometry[:, 1] *= scale[1]

    return patches.Polygon(
        geometry,
        linewidth=linewidth,
        edgecolor=(*color, alpha),
        facecolor=(*color, alpha) if fill else "none",
        fill=fill,
        label=label,
    )


def create_obj_patch(
    geometry: Union[BoundingBox, Polygon4P, np.ndarray],
    page_dimensions: tuple[int, int],
    **kwargs: Any,
) -> patches.Patch:
    """Create an appropriate patch (rectangle or polygon) based on geometry."""
    if isinstance(geometry, tuple):
        if len(geometry) == 2:
            return rect_patch(geometry, page_dimensions, **kwargs)
        elif len(geometry) == 4:
            return polygon_patch(np.asarray(geometry), page_dimensions, **kwargs)
    elif isinstance(geometry, np.ndarray) and geometry.shape == (4, 2):
        return polygon_patch(geometry, page_dimensions, **kwargs)

    raise ValueError("Unsupported geometry format. Must be 2-point or 4-point.")


def get_colors(num_colors: int) -> list[tuple[float, float, float]]:
    """Generate distinguishable RGB colors in [0, 1] range."""
    return [
        colorsys.hls_to_rgb(
            hue / 360.0,
            (50 + np.random.rand() * 10) / 100.0,
            (90 + np.random.rand() * 10) / 100.0,
        )
        for hue in np.linspace(0.0, 360.0, num_colors, endpoint=False)
    ]

# ------------------------ Visualization Functions ------------------------ #

def visualize_page(
    page: dict[str, Any],
    image: np.ndarray,
    words_only: bool = True,
    display_artefacts: bool = True,
    scale: float = 10,
    interactive: bool = True,
    add_labels: bool = True,
    **kwargs: Any,
) -> Figure:
    """Visualize OCR predictions on a full page."""
    h, w = image.shape[:2]
    fig_size = (scale * w / h, scale) if h > w else (scale, h / w * scale)
    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(image)
    ax.axis("off")

    artists = [] if interactive else None

    for block in page.get("blocks", []):
        if not words_only:
            block_patch = create_obj_patch(
                block["geometry"], page["dimensions"], label="block", color=(0, 1, 0), linewidth=1, **kwargs
            )
            ax.add_patch(block_patch)
            if artists is not None:
                artists.append(block_patch)

        for line in block.get("lines", []):
            if not words_only:
                line_patch = create_obj_patch(
                    line["geometry"], page["dimensions"], label="line", color=(1, 0, 0), linewidth=1, **kwargs
                )
                ax.add_patch(line_patch)
                if artists is not None:
                    artists.append(line_patch)

            for word in line.get("words", []):
                word_patch = create_obj_patch(
                    word["geometry"],
                    page["dimensions"],
                    label=f"{word['value']} (confidence: {word['confidence']:.2%})",
                    color=(0, 0, 1),
                    **kwargs,
                )
                ax.add_patch(word_patch)
                if artists is not None:
                    artists.append(word_patch)
                elif add_labels:
                    if isinstance(word["geometry"], list) and len(word["geometry"]) == 2:
                        x, y = word["geometry"][0]
                        ax.text(
                            x * page["dimensions"][1],
                            y * page["dimensions"][0],
                            word["value"],
                            size=10,
                            alpha=0.5,
                            color="blue",
                        )

        if display_artefacts:
            for artefact in block.get("artefacts", []):
                artefact_patch = create_obj_patch(
                    artefact["geometry"],
                    page["dimensions"],
                    label="artefact",
                    color=(0.5, 0.5, 0.5),
                    linewidth=1,
                    **kwargs,
                )
                ax.add_patch(artefact_patch)
                if artists is not None:
                    artists.append(artefact_patch)

    if interactive and artists:
        import mplcursors
        mplcursors.Cursor(artists, hover=2).connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))

    fig.tight_layout(pad=0.0)
    return fig


def visualize_kie_page(
    page: dict[str, Any],
    image: np.ndarray,
    words_only: bool = False,
    display_artefacts: bool = True,
    scale: float = 10,
    interactive: bool = True,
    add_labels: bool = True,
    **kwargs: Any,
) -> Figure:
    """Visualize Key Information Extraction predictions on a full page."""
    h, w = image.shape[:2]
    fig_size = (scale * w / h, scale) if h > w else (scale, h / w * scale)
    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(image)
    ax.axis("off")

    artists = [] if interactive else None
    color_map = {key: color for color, key in zip(get_colors(len(page["predictions"])), page["predictions"])}

    for label_key, predictions in page["predictions"].items():
        for pred in predictions:
            patch = create_obj_patch(
                pred["geometry"],
                page["dimensions"],
                label=f"{label_key} \n{pred['value']} (confidence: {pred['confidence']:.2%})",
                color=color_map[label_key],
                linewidth=1,
                **kwargs,
            )
            ax.add_patch(patch)
            if artists is not None:
                artists.append(patch)

    if interactive and artists:
        import mplcursors
        mplcursors.Cursor(artists, hover=2).connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))

    fig.tight_layout(pad=0.0)
    return fig

# ------------------------ Utility Drawing Function ------------------------ #

def draw_boxes(
    boxes: np.ndarray,
    image: np.ndarray,
    color: tuple[int, int, int] | None = None,
    **kwargs: Any,
) -> None:
    """Draw an array of straight bounding boxes (in relative coords) on an image."""
    h, w = image.shape[:2]
    abs_boxes = deepcopy(boxes)
    abs_boxes[:, [0, 2]] *= w
    abs_boxes[:, [1, 3]] *= h
    abs_boxes = abs_boxes.astype(np.int32)

    for xmin, ymin, xmax, ymax in abs_boxes:
        cv2.rectangle(
            image,
            (xmin, ymin),
            (xmax, ymax),
            color=color if color else (0, 0, 255),
            thickness=2,
        )

    plt.imshow(image)
    plt.plot(**kwargs)
