"""Table detection helpers built on OpenCV morphological operations."""

from __future__ import annotations

import logging
from typing import List

import cv2
import numpy as np

from models import CellBBox

LOGGER = logging.getLogger(__name__)


def detect_table_and_cells(
    image_path: str,
    min_table_area_ratio: float = 0.2,
    min_line_length_ratio: float = 0.3,
) -> List[CellBBox]:
    """Identify the dominant table grid in ``image_path`` and return its cell boxes."""

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Convert to grayscale + threshold so horizontal/vertical lines have strong contrast.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    height, width = binary.shape
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(10, width // 50), 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(10, height // 40))
    )

    horizontal_lines = cv2.erode(binary, horizontal_kernel, iterations=2)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=2)

    vertical_lines = cv2.erode(binary, vertical_kernel, iterations=2)
    vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=2)

    # Combine the line masks to create a table grid mask and find the largest contour.
    grid = cv2.bitwise_or(horizontal_lines, vertical_lines)
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No table-like contour detected")

    table_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(table_contour)
    area_ratio = (w * h) / (width * height)
    if area_ratio < min_table_area_ratio:
        raise ValueError("Largest contour is too small to be the main table")

    roi_horizontal = horizontal_lines[y : y + h, x : x + w]
    roi_vertical = vertical_lines[y : y + h, x : x + w]

    row_boundaries = _grid_boundaries(
        roi_horizontal,
        axis=1,
        offset=y,
        min_line_length_ratio=min_line_length_ratio,
        fallback_start=y,
        fallback_end=y + h,
    )
    col_boundaries = _grid_boundaries(
        roi_vertical,
        axis=0,
        offset=x,
        min_line_length_ratio=min_line_length_ratio,
        fallback_start=x,
        fallback_end=x + w,
    )

    cells: List[CellBBox] = []
    for row_idx in range(len(row_boundaries) - 1):
        # Build cells for this row while allowing horizontally merged cells.
        row_cells = _build_row_cells_with_merges(
            row_idx=row_idx,
            row_boundaries=row_boundaries,
            col_boundaries=col_boundaries,
            roi_vertical=roi_vertical,
            table_x=x,
            table_y=y,
        )
        cells.extend(row_cells)

    if not cells:
        LOGGER.warning("Failed to derive grid; falling back to single table crop")
        cells.append(
            CellBBox(
                row_index=0,
                column_index=0,
                x_min=int(x),
                y_min=int(y),
                x_max=int(x + w),
                y_max=int(y + h),
            )
        )

    return cells


def _build_row_cells_with_merges(
    row_idx: int,
    row_boundaries: List[int],
    col_boundaries: List[int],
    roi_vertical: np.ndarray,
    table_x: int,
    table_y: int,
    merge_threshold_ratio: float = 0.3,
) -> List[CellBBox]:
    """Construct cells for a single row, merging horizontally where lines are weak.

    The global column boundaries are derived from vertical lines aggregated over
    the whole table. In header/footer rows, some of those lines may be absent,
    which corresponds to visually merged cells. Here we re-merge adjacent
    columns when there is little or no vertical line energy at that boundary
    for the current row.
    """

    y_min = row_boundaries[row_idx]
    y_max = row_boundaries[row_idx + 1]
    row_height = y_max - y_min
    if row_height < 5:
        return []

    # If this is the last row of the table, treat it as a single merged footer cell.
    # Borehole logs often have a legend/notes row at the bottom that visually spans
    # the full width even though column grid lines continue underneath.
    if row_idx == len(row_boundaries) - 2:
        x_min_footer = col_boundaries[0]
        x_max_footer = col_boundaries[-1]
        if (x_max_footer - x_min_footer) >= 5 and row_height >= 5:
            return [
                CellBBox(
                    row_index=row_idx,
                    column_index=0,
                    x_min=int(x_min_footer),
                    y_min=int(y_min),
                    x_max=int(x_max_footer),
                    y_max=int(y_max),
                )
            ]

    # Map the row span into the ROI coordinates used by roi_vertical.
    roi_y0 = max(0, y_min - table_y)
    roi_y1 = min(roi_vertical.shape[0], y_max - table_y)
    if roi_y1 <= roi_y0:
        return []

    n_cols = len(col_boundaries) - 1
    if n_cols <= 0:
        return []

    # Decide where vertical boundaries are really present for this row.
    has_boundary = [False] * (n_cols - 1)
    for idx in range(n_cols - 1):
        boundary_x = col_boundaries[idx + 1]
        roi_x = boundary_x - table_x
        x0 = max(0, roi_x - 1)
        x1 = min(roi_vertical.shape[1], roi_x + 2)
        if x1 <= x0:
            continue
        strip = roi_vertical[roi_y0:roi_y1, x0:x1]
        line_strength = int((strip > 0).sum())
        # Consider a boundary "present" if enough pixels are set relative to row height.
        if line_strength >= merge_threshold_ratio * row_height:
            has_boundary[idx] = True

    # Merge adjacent columns where there is no boundary for this row.
    row_cells: List[CellBBox] = []
    segment_start = 0
    segment_idx = 0
    for idx in range(n_cols - 1):
        if has_boundary[idx]:
            x_min = col_boundaries[segment_start]
            x_max = col_boundaries[idx + 1]
            if (x_max - x_min) >= 5 and row_height >= 5:
                row_cells.append(
                    CellBBox(
                        row_index=row_idx,
                        column_index=segment_idx,
                        x_min=int(x_min),
                        y_min=int(y_min),
                        x_max=int(x_max),
                        y_max=int(y_max),
                    )
                )
                segment_idx += 1
            segment_start = idx + 1

    # Final segment (may span multiple grid columns if no inner boundaries).
    if segment_start <= n_cols - 1:
        x_min = col_boundaries[segment_start]
        x_max = col_boundaries[-1]
        if (x_max - x_min) >= 5 and row_height >= 5:
            row_cells.append(
                CellBBox(
                    row_index=row_idx,
                    column_index=segment_idx,
                    x_min=int(x_min),
                    y_min=int(y_min),
                    x_max=int(x_max),
                    y_max=int(y_max),
                )
            )

    return row_cells


def _grid_boundaries(
    line_image: np.ndarray,
    axis: int,
    offset: int,
    min_line_length_ratio: float,
    fallback_start: int,
    fallback_end: int,
) -> List[int]:
    """Collapse a line mask along ``axis`` to find row/column separators."""

    projection = np.sum(line_image > 0, axis=axis)
    max_val = projection.max()
    if max_val == 0:
        return [fallback_start, fallback_end]

    threshold = max_val * min_line_length_ratio
    segments = []
    start_idx = None
    for idx, value in enumerate(projection):
        if value >= threshold:
            if start_idx is None:
                start_idx = idx
        elif start_idx is not None:
            segments.append((start_idx, idx))
            start_idx = None
    if start_idx is not None:
        segments.append((start_idx, len(projection)))

    centers = [int((s + e) / 2) + offset for s, e in segments]

    boundaries = [fallback_start]
    boundaries.extend(sorted(set(centers)))
    boundaries.append(fallback_end)

    deduped = []
    for value in boundaries:
        if not deduped or value - deduped[-1] > 2:
            deduped.append(value)
        else:
            deduped[-1] = value

    return deduped
