"""Shared dataclasses and type aliases for the borehole OCR pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

RangeMatch = Tuple[Optional[float], Optional[float]]


@dataclass(frozen=True)
class CellBBox:
    """Represents a single table cell bounding box in page pixel coordinates."""

    row_index: int
    column_index: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int
