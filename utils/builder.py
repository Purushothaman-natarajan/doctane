from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fclusterdata

from doctane.utils.io_elements import Block, Document, KIEDocument, KIEPage, Line, Page, Prediction, Word
from doctane.utils.geometry import estimate_page_angle, resolve_enclosing_bbox, resolve_enclosing_rbbox, rotate_boxes
from doctane.utils.representation import NestedObject

__all__ = ["DocumentBuilder"]


class DocumentBuilder(NestedObject):
    """
    Builds a document structure by grouping detected OCR words into lines and blocks.

    Args:
        resolve_lines (bool): Whether to group words into lines.
        resolve_blocks (bool): Whether to group lines into blocks.
        paragraph_break (float): Threshold ratio of horizontal space between words to break into paragraphs.
        export_as_straight_boxes (bool): If True, rotated boxes will be converted to straight bounding boxes.
    """

    def __init__(
        self,
        resolve_lines: bool = True,
        resolve_blocks: bool = False,
        paragraph_break: float = 0.035,
        export_as_straight_boxes: bool = False,
    ) -> None:
        self.resolve_lines = resolve_lines
        self.resolve_blocks = resolve_blocks
        self.paragraph_break = paragraph_break
        self.export_as_straight_boxes = export_as_straight_boxes

    @staticmethod
    def _sort_boxes(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Sort boxes top-to-bottom, left-to-right. If rotated boxes are detected,
        straighten them before sorting.

        Args:
            boxes (np.ndarray): Array of shape (N, 4) or (N, 4, 2).

        Returns:
            Tuple of:
                - Sorted indices of shape (N,)
                - Possibly straightened boxes of shape (N, 4)
        """
        if boxes.ndim == 3:
            boxes = rotate_boxes(
                loc_preds=boxes,
                angle=-estimate_page_angle(boxes),
                orig_shape=(1024, 1024),
                min_angle=5.0,
            )
            boxes = np.concatenate((boxes.min(1), boxes.max(1)), -1)
        return (boxes[:, 0] + 2 * boxes[:, 3] / np.median(boxes[:, 3] - boxes[:, 1])).argsort(), boxes

    def _resolve_sub_lines(self, boxes: np.ndarray, word_idcs: list[int]) -> list[list[int]]:
        """
        Optionally split a horizontal line into sub-lines based on horizontal spacing.

        Args:
            boxes (np.ndarray): Straight boxes of shape (N, 4).
            word_idcs (list[int]): Indexes of the words in the current line.

        Returns:
            List of sub-lines (each a list of word indices).
        """
        lines = []
        # Sort words horizontally
        word_idcs = [word_idcs[idx] for idx in boxes[word_idcs, 0].argsort().tolist()]

        # Eventually split line horizontally
        if len(word_idcs) < 2:
            lines.append(word_idcs)
        else:
            sub_line = [word_idcs[0]]
            for i in word_idcs[1:]:
                horiz_break = True

                prev_box = boxes[sub_line[-1]]
                # Compute distance between boxes
                dist = boxes[i, 0] - prev_box[2]
                # If distance between boxes is lower than paragraph break, same sub-line
                if dist < self.paragraph_break:
                    horiz_break = False

                if horiz_break:
                    lines.append(sub_line)
                    sub_line = []

                sub_line.append(i)
            lines.append(sub_line)

        return lines

    def _resolve_lines(self, boxes: np.ndarray) -> list[list[int]]:
        """
        Group boxes into lines based on vertical overlap and height heuristics.

        Args:
            boxes (np.ndarray): Boxes of shape (N, 4) or (N, 4, 2).

        Returns:
            List of lines (each a list of word indices).
        """
        # Sort boxes, and straighten the boxes if they are rotated
        idxs, boxes = self._sort_boxes(boxes)

        # Compute median for boxes heights
        y_med = np.median(boxes[:, 3] - boxes[:, 1])

        lines = []
        words = [idxs[0]]  # Assign the top-left word to the first line
        # Define a mean y-center for the line
        y_center_sum = boxes[idxs[0]][[1, 3]].mean()

        for idx in idxs[1:]:
            vert_break = True

            # Compute y_dist
            y_dist = abs(boxes[idx][[1, 3]].mean() - y_center_sum / len(words))
            # If y-center of the box is close enough to mean y-center of the line, same line
            if y_dist < y_med / 2:
                vert_break = False

            if vert_break:
                # Compute sub-lines (horizontal split)
                lines.extend(self._resolve_sub_lines(boxes, words))
                words = []
                y_center_sum = 0

            words.append(idx)
            y_center_sum += boxes[idx][[1, 3]].mean()

        # Use the remaining words to form the last(s) line(s)
        if len(words) > 0:
            # Compute sub-lines (horizontal split)
            lines.extend(self._resolve_sub_lines(boxes, words))

        return lines

    @staticmethod
    def _resolve_blocks(boxes: np.ndarray, lines: list[list[int]]) -> list[list[list[int]]]:
        """
        Group lines into blocks using hierarchical clustering on vertical position.

        Args:
            boxes (np.ndarray): Boxes of shape (N, 4) or (N, 4, 2).

        Returns:
            List of blocks (each a list of line indices).
        """
        # Resolve enclosing boxes of lines
        if boxes.ndim == 3:
            box_lines: np.ndarray = np.asarray([
                resolve_enclosing_rbbox([tuple(boxes[idx, :, :]) for idx in line])  # type: ignore[misc]
                for line in lines
            ])
        else:
            _box_lines = [
                resolve_enclosing_bbox([(tuple(boxes[idx, :2]), tuple(boxes[idx, 2:])) for idx in line])
                for line in lines
            ]
            box_lines = np.asarray([(x1, y1, x2, y2) for ((x1, y1), (x2, y2)) in _box_lines])

        # Compute geometrical features of lines to clusterize
        # Clusterizing only with box centers yield to poor results for complex documents
        if boxes.ndim == 3:
            box_features: np.ndarray = np.stack(
                (
                    (box_lines[:, 0, 0] + box_lines[:, 0, 1]) / 2,
                    (box_lines[:, 0, 0] + box_lines[:, 2, 0]) / 2,
                    (box_lines[:, 0, 0] + box_lines[:, 2, 1]) / 2,
                    (box_lines[:, 0, 1] + box_lines[:, 2, 1]) / 2,
                    (box_lines[:, 0, 1] + box_lines[:, 2, 0]) / 2,
                    (box_lines[:, 2, 0] + box_lines[:, 2, 1]) / 2,
                ),
                axis=-1,
            )
        else:
            box_features = np.stack(
                (
                    (box_lines[:, 0] + box_lines[:, 3]) / 2,
                    (box_lines[:, 1] + box_lines[:, 2]) / 2,
                    (box_lines[:, 0] + box_lines[:, 2]) / 2,
                    (box_lines[:, 1] + box_lines[:, 3]) / 2,
                    box_lines[:, 0],
                    box_lines[:, 1],
                ),
                axis=-1,
            )
        # Compute clusters
        clusters = fclusterdata(box_features, t=0.1, depth=4, criterion="distance", metric="euclidean")

        _blocks: dict[int, list[int]] = {}
        # Form clusters
        for line_idx, cluster_idx in enumerate(clusters):
            if cluster_idx in _blocks.keys():
                _blocks[cluster_idx].append(line_idx)
            else:
                _blocks[cluster_idx] = [line_idx]

        # Retrieve word-box level to return a fully nested structure
        blocks = [[lines[idx] for idx in block] for block in _blocks.values()]

        return blocks

    def _build_blocks(
        self,
        boxes: np.ndarray,
        objectness_scores: np.ndarray,
        word_preds: list[tuple[str, float]],
        crop_orientations: list[dict[str, Any]],
    ) -> list[Block]:
        """Gather independent words in structured blocks

        Args:
            boxes: bounding boxes of all detected words of the page, of shape (N, 4) or (N, 4, 2)
            objectness_scores: objectness scores of all detected words of the page, of shape N
            word_preds: list of all detected words of the page, of shape N
            crop_orientations: list of dictoinaries containing
                the general orientation (orientations + confidences) of the crops

        Returns:
            list of block elements
        """
        if boxes.shape[0] != len(word_preds):
            raise ValueError(f"Incompatible argument lengths: {boxes.shape[0]}, {len(word_preds)}")

        if boxes.shape[0] == 0:
            return []

        # Decide whether we try to form lines
        _boxes = boxes
        if self.resolve_lines:
            lines = self._resolve_lines(_boxes if _boxes.ndim == 3 else _boxes[:, :4])
            # Decide whether we try to form blocks
            if self.resolve_blocks and len(lines) > 1:
                _blocks = self._resolve_blocks(_boxes if _boxes.ndim == 3 else _boxes[:, :4], lines)
            else:
                _blocks = [lines]
        else:
            # Sort bounding boxes, one line for all boxes, one block for the line
            lines = [self._sort_boxes(_boxes if _boxes.ndim == 3 else _boxes[:, :4])[0]]  # type: ignore[list-item]
            _blocks = [lines]

        blocks = [
            Block([
                Line([
                    Word(
                        *word_preds[idx],
                        tuple(tuple(pt) for pt in boxes[idx].tolist()),  # type: ignore[arg-type]
                        float(objectness_scores[idx]),
                        crop_orientations[idx],
                    )
                    if boxes.ndim == 3
                    else Word(
                        *word_preds[idx],
                        ((boxes[idx, 0], boxes[idx, 1]), (boxes[idx, 2], boxes[idx, 3])),
                        float(objectness_scores[idx]),
                        crop_orientations[idx],
                    )
                    for idx in line
                ])
                for line in lines
            ])
            for lines in _blocks
        ]

        return blocks

    def extra_repr(self) -> str:
        return (
            f"resolve_lines={self.resolve_lines}, resolve_blocks={self.resolve_blocks}, "
            f"paragraph_break={self.paragraph_break}, "
            f"export_as_straight_boxes={self.export_as_straight_boxes}"
        )

    def __call__(
        self,
        pages: list[np.ndarray],
        boxes: list[np.ndarray],
        objectness_scores: list[np.ndarray],
        text_preds: list[list[tuple[str, float]]],
        page_shapes: list[tuple[int, int]],
        crop_orientations: list[dict[str, Any]],
        orientations: list[dict[str, Any]] | None = None,
        languages: list[dict[str, Any]] | None = None,
    ) -> Document:
        """Re-arrange detected words into structured blocks

        Args:
            pages: list of N elements, where each element represents the page image
            boxes: list of N elements, where each element represents the localization predictions, of shape (*, 4)
                or (*, 4, 2) for all words for a given page
            objectness_scores: list of N elements, where each element represents the objectness scores
            text_preds: list of N elements, where each element is the list of all word prediction (text + confidence)
            page_shapes: shape of each page, of size N
            crop_orientations: list of N elements, where each element is
                a dictionary containing the general orientation (orientations + confidences) of the crops
            orientations: optional, list of N elements,
                where each element is a dictionary containing the orientation (orientation + confidence)
            languages: optional, list of N elements,
                where each element is a dictionary containing the language (language + confidence)

        Returns:
            document object
        """
        if len(boxes) != len(text_preds) != len(crop_orientations) != len(objectness_scores) or len(boxes) != len(
            page_shapes
        ) != len(crop_orientations) != len(objectness_scores):
            raise ValueError("All arguments are expected to be lists of the same size")

        _orientations = (
            orientations if isinstance(orientations, list) else [None] * len(boxes)  # type: ignore[list-item]
        )
        _languages = languages if isinstance(languages, list) else [None] * len(boxes)  # type: ignore[list-item]
        if self.export_as_straight_boxes and len(boxes) > 0:
            # If boxes are already straight OK, else fit a bounding rect
            if boxes[0].ndim == 3:
                # Iterate over pages and boxes
                boxes = [np.concatenate((p_boxes.min(1), p_boxes.max(1)), 1) for p_boxes in boxes]

        _pages = [
            Page(
                page,
                self._build_blocks(
                    page_boxes,
                    loc_scores,
                    word_preds,
                    word_crop_orientations,
                ),
                _idx,
                shape,
                orientation,
                language,
            )
            for page, _idx, shape, page_boxes, loc_scores, word_preds, word_crop_orientations, orientation, language in zip(  # noqa: E501
                pages,
                range(len(boxes)),
                page_shapes,
                boxes,
                objectness_scores,
                text_preds,
                crop_orientations,
                _orientations,
                _languages,
            )
        ]

        return Document(_pages)


class KIEDocumentBuilder(DocumentBuilder):
    """Implements a KIE document builder

    Args:
        resolve_lines: whether words should be automatically grouped into lines
        resolve_blocks: whether lines should be automatically grouped into blocks
        paragraph_break: relative length of the minimum space separating paragraphs
        export_as_straight_boxes: if True, force straight boxes in the export (fit a rectangle
            box to all rotated boxes). Else, keep the boxes format unchanged, no matter what it is.
    """

    def __call__(  # type: ignore[override]
        self,
        pages: list[np.ndarray],
        boxes: list[dict[str, np.ndarray]],
        objectness_scores: list[dict[str, np.ndarray]],
        text_preds: list[dict[str, list[tuple[str, float]]]],
        page_shapes: list[tuple[int, int]],
        crop_orientations: list[dict[str, list[dict[str, Any]]]],
        orientations: list[dict[str, Any]] | None = None,
        languages: list[dict[str, Any]] | None = None,
    ) -> KIEDocument:
        """Re-arrange detected words into structured predictions

        Args:
            pages: list of N elements, where each element represents the page image
            boxes: list of N dictionaries, where each element represents the localization predictions for a class,
                of shape (*, 5) or (*, 6) for all predictions
            objectness_scores: list of N dictionaries, where each element represents the objectness scores for a class
            text_preds: list of N dictionaries, where each element is the list of all word prediction
            page_shapes: shape of each page, of size N
            crop_orientations: list of N dictonaries, where each element is
                a list containing the general crop orientations (orientations + confidences) of the crops
            orientations: optional, list of N elements,
                where each element is a dictionary containing the orientation (orientation + confidence)
            languages: optional, list of N elements,
                where each element is a dictionary containing the language (language + confidence)

        Returns:
            document object
        """
        if len(boxes) != len(text_preds) != len(crop_orientations) != len(objectness_scores) or len(boxes) != len(
            page_shapes
        ) != len(crop_orientations) != len(objectness_scores):
            raise ValueError("All arguments are expected to be lists of the same size")
        _orientations = (
            orientations if isinstance(orientations, list) else [None] * len(boxes)  # type: ignore[list-item]
        )
        _languages = languages if isinstance(languages, list) else [None] * len(boxes)  # type: ignore[list-item]
        if self.export_as_straight_boxes and len(boxes) > 0:
            # If boxes are already straight OK, else fit a bounding rect
            if next(iter(boxes[0].values())).ndim == 3:
                straight_boxes: list[dict[str, np.ndarray]] = []
                # Iterate over pages
                for p_boxes in boxes:
                    # Iterate over boxes of the pages
                    straight_boxes_dict = {}
                    for k, box in p_boxes.items():
                        straight_boxes_dict[k] = np.concatenate((box.min(1), box.max(1)), 1)
                    straight_boxes.append(straight_boxes_dict)
                boxes = straight_boxes

        _pages = [
            KIEPage(
                page,
                {
                    k: self._build_blocks(
                        page_boxes[k],
                        loc_scores[k],
                        word_preds[k],
                        word_crop_orientations[k],
                    )
                    for k in page_boxes.keys()
                },
                _idx,
                shape,
                orientation,
                language,
            )
            for page, _idx, shape, page_boxes, loc_scores, word_preds, word_crop_orientations, orientation, language in zip(  # noqa: E501
                pages,
                range(len(boxes)),
                page_shapes,
                boxes,
                objectness_scores,
                text_preds,
                crop_orientations,
                _orientations,
                _languages,
            )
        ]

        return KIEDocument(_pages)

    def _build_blocks(  # type: ignore[override]
        self,
        boxes: np.ndarray,
        objectness_scores: np.ndarray,
        word_preds: list[tuple[str, float]],
        crop_orientations: list[dict[str, Any]],
    ) -> list[Prediction]:
        """Gather independent words in structured blocks

        Args:
            boxes: bounding boxes of all detected words of the page, of shape (N, 4) or (N, 4, 2)
            objectness_scores: objectness scores of all detected words of the page
            word_preds: list of all detected words of the page, of shape N
            crop_orientations: list of orientations for each word crop

        Returns:
            list of block elements
        """
        if boxes.shape[0] != len(word_preds):
            raise ValueError(f"Incompatible argument lengths: {boxes.shape[0]}, {len(word_preds)}")

        if boxes.shape[0] == 0:
            return []

        # Decide whether we try to form lines
        _boxes = boxes
        idxs, _ = self._sort_boxes(_boxes if _boxes.ndim == 3 else _boxes[:, :4])
        predictions = [
            Prediction(
                value=word_preds[idx][0],
                confidence=word_preds[idx][1],
                geometry=tuple(tuple(pt) for pt in boxes[idx].tolist()),  # type: ignore[arg-type]
                objectness_score=float(objectness_scores[idx]),
                crop_orientation=crop_orientations[idx],
            )
            if boxes.ndim == 3
            else Prediction(
                value=word_preds[idx][0],
                confidence=word_preds[idx][1],
                geometry=((boxes[idx, 0], boxes[idx, 1]), (boxes[idx, 2], boxes[idx, 3])),
                objectness_score=float(objectness_scores[idx]),
                crop_orientation=crop_orientations[idx],
            )
            for idx in idxs
        ]
        return predictions




# ### add the following to support column and row detection using models ;:

# def _resolve_lines_from_mask(self, mask: np.ndarray, boxes: np.ndarray) -> list[list[int]]:
#     """Assign words to lines based on overlap with row-wise mask labels."""
#     h, w = mask.shape
#     line_ids = []

#     for box in boxes:
#         poly = np.round(box).astype(np.int32)
#         mask_crop = mask[poly[:,1].min():poly[:,1].max()+1, poly[:,0].min():poly[:,0].max()+1]
#         if mask_crop.size == 0:
#             line_ids.append(-1)
#         else:
#             unique, counts = np.unique(mask_crop, return_counts=True)
#             line_ids.append(int(unique[np.argmax(counts)]))

#     lines = defaultdict(list)
#     for idx, line_id in enumerate(line_ids):
#         if line_id >= 0:
#             lines[line_id].append(idx)
#     return list(lines.values())


# def _resolve_blocks_from_mask(self, mask: np.ndarray, lines: list[Line]) -> list[list[int]]:
#     """Group lines into blocks using column-wise mask."""
#     blocks = defaultdict(list)

#     for i, line in enumerate(lines):
#         box = np.round(line.box).astype(np.int32)
#         x_min, y_min = box[:,0].min(), box[:,1].min()
#         x_max, y_max = box[:,0].max(), box[:,1].max()
#         patch = mask[y_min:y_max+1, x_min:x_max+1]
#         if patch.size == 0:
#             continue
#         labels, counts = np.unique(patch, return_counts=True)
#         label = labels[np.argmax(counts)]
#         blocks[int(label)].append(i)

#     return list(blocks.values())


# if self.row_mask_model is not None:
#     image = pred.image  # assumed to be available in prediction
#     row_mask = self.row_mask_model(image)

#     lines = self._resolve_lines_from_mask(row_mask, boxes)
# else:
#     lines = self._resolve_lines(boxes) if self.resolve_lines else [[i] for i in range(len(pred.words))]


# if self.col_mask_model is not None:
#     col_mask = self.col_mask_model(image)
#     blocks = self._resolve_blocks_from_mask(col_mask, word_lines)
# else:
#     line_boxes = np.stack([line.box for line in word_lines])
#     blocks = self._resolve_blocks(line_boxes) if self.resolve_blocks else None
