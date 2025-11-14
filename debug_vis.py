"""Debug visualisation helpers for table/cell detection.

This module is optional and used to produce annotated page images that show
where the table grid and cells were detected. These images make it easier
to diagnose misaligned grids or badly segmented rows/columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple

import cv2

from models import CellBBox


def draw_cells_overlay(
    page_image_path: str,
    cell_bboxes: Sequence[CellBBox],
    output_path: str,
    text_records: Optional[Sequence[Dict[str, object]]] = None,
) -> str:
    """Draw cell bounding boxes on a copy of ``page_image_path`` and save it.

    The output image highlights each detected cell with a green rectangle and
    a small label indicating its row/column index (e.g. ``r3c1``). This is
    intended for manual inspection rather than for further automated steps.
    """

    image = cv2.imread(page_image_path)
    if image is None:
        raise FileNotFoundError(
            f"Could not read page image for debug overlay: {page_image_path}"
        )

    overlay = image.copy()

    # Map (row_index, column_index) -> text bounding box in page coordinates.
    text_boxes: Dict[Tuple[int, int], Dict[str, int]] = {}
    if text_records is not None:
        for rec in text_records:
            try:
                row_idx = int(rec.get("row_index"))  # type: ignore[arg-type]
                col_idx = int(rec.get("column_index"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue

            text = str(rec.get("text", "") or "").strip()
            bbox_dict = rec.get("text_bbox")
            if not text or not isinstance(bbox_dict, dict):
                continue

            try:
                x_min = int(bbox_dict["x_min"])
                y_min = int(bbox_dict["y_min"])
                x_max = int(bbox_dict["x_max"])
                y_max = int(bbox_dict["y_max"])
            except (KeyError, TypeError, ValueError):
                continue

            text_boxes[(row_idx, col_idx)] = {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            }

    for bbox in cell_bboxes:
        top_left = (int(bbox.x_min), int(bbox.y_min))
        bottom_right = (int(bbox.x_max), int(bbox.y_max))

        # Base green rectangle for every detected cell (table structure).
        cv2.rectangle(overlay, top_left, bottom_right, (0, 255, 0), 2)

        # If OCR produced a text bounding box for this cell, draw that region in red.
        tb = text_boxes.get((bbox.row_index, bbox.column_index))
        if tb is not None:
            red_top_left = (tb["x_min"], tb["y_min"])
            red_bottom_right = (tb["x_max"], tb["y_max"])
            cv2.rectangle(overlay, red_top_left, red_bottom_right, (0, 0, 255), 2)

        label = f"r{bbox.row_index}c{bbox.column_index}"
        label_pos = (top_left[0] + 2, top_left[1] + 12)
        cv2.putText(
            overlay,
            label,
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
            lineType=cv2.LINE_AA,
        )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)

    return str(out_path)
