"""Post-processing utilities for depth parsing and CSV export."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from models import RangeMatch

LOGGER = logging.getLogger(__name__)

# Regex helpers used to interpret numeric depth ranges within OCR text.
DEPTH_RANGE_PATTERN = re.compile(
    r"(?P<from>\d+(?:[\.,]\d+)?)\s*(?:-|to|–|—)\s*(?P<to>\d+(?:[\.,]\d+)?)",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"\d+(?:[\.,]\d+)?")


def postprocess_and_export(
    records: Sequence[Dict[str, object]],
    output_csv_path: str,
    depth_columns: Sequence[int] = (0, 1),
    description_columns: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Aggregate cell-level OCR ``records`` into one row per depth interval."""

    df = pd.DataFrame(records)
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        LOGGER.warning("No OCR records to export; writing empty CSV")
        empty_df = pd.DataFrame(
            columns=[
                "file_name",
                "page_index",
                "row_index",
                "depth_from_m",
                "depth_to_m",
                "description",
                "raw_cells_json",
            ]
        )
        empty_df.to_csv(output_path, index=False)
        return empty_df

    grouped: List[Dict[str, object]] = []
    for (file_name, page_index, row_index), group in df.groupby(
        ["file_name", "page_index", "row_index"]
    ):
        sorted_group = group.sort_values("column_index")
        column_text = {
            int(row.column_index): str(row.text).strip()
            for row in sorted_group.itertuples()
            if str(row.text).strip()
        }
        depth_from, depth_to = _extract_depths(column_text, depth_columns)
        description = _build_description(column_text, depth_columns, description_columns)

        grouped.append(
            {
                "file_name": file_name,
                "page_index": page_index,
                "row_index": row_index,
                "depth_from_m": depth_from,
                "depth_to_m": depth_to,
                "description": description,
                "raw_cells_json": json.dumps(column_text, ensure_ascii=False),
            }
        )

    result_df = pd.DataFrame(grouped)
    result_df.to_csv(output_path, index=False)
    LOGGER.info("Exported %d intervals to %s", len(result_df), output_path)
    return result_df


def compare_with_manual(
    auto_csv_path: str,
    manual_csv_path: str,
    id_columns: Sequence[str] = ("file_name", "page_index", "row_index"),
    depth_tolerance: float = 0.05,
) -> pd.DataFrame:
    """Compare automatic output to a manual reference CSV, flagging mismatches."""

    auto_df = pd.read_csv(auto_csv_path)
    manual_df = pd.read_csv(manual_csv_path)

    merged = auto_df.merge(
        manual_df,
        on=list(id_columns),
        how="outer",
        suffixes=("_auto", "_manual"),
        indicator=True,
    )

    def _is_close(a: Optional[float], b: Optional[float]) -> bool:
        if pd.isna(a) or pd.isna(b):
            return False
        return abs(float(a) - float(b)) <= depth_tolerance

    mismatches = []
    for row in merged.itertuples():
        if row._merge != "both":
            mismatches.append(row)
            continue
        if not _is_close(row.depth_from_m_auto, row.depth_from_m_manual) or not _is_close(
            row.depth_to_m_auto, row.depth_to_m_manual
        ):
            mismatches.append(row)
            continue
        if str(row.description_auto).strip() != str(row.description_manual).strip():
            mismatches.append(row)

    return pd.DataFrame(mismatches)


def _extract_depths(
    column_text: Dict[int, str],
    depth_columns: Sequence[int],
) -> RangeMatch:
    """Parse numeric depth information from the configured table columns."""

    texts = [column_text.get(col, "") for col in depth_columns]
    combined = " ".join(filter(None, texts))

    match = DEPTH_RANGE_PATTERN.search(combined)
    if match:
        return (_to_float(match.group("from")), _to_float(match.group("to")))

    numbers = [
        _to_float(num)
        for text in texts
        for num in NUMBER_PATTERN.findall(text)
        if _to_float(num) is not None
    ]
    if len(numbers) >= 2:
        return numbers[0], numbers[1]
    if len(numbers) == 1:
        return numbers[0], None

    for text in column_text.values():
        match = DEPTH_RANGE_PATTERN.search(text)
        if match:
            return (_to_float(match.group("from")), _to_float(match.group("to")))

    return None, None


def _build_description(
    column_text: Dict[int, str],
    depth_columns: Sequence[int],
    description_columns: Optional[Sequence[int]],
) -> str:
    """Join together the text columns that aren't used for depth values."""

    if description_columns is None:
        description_columns = [col for col in column_text if col not in depth_columns]

    texts = [column_text.get(col, "") for col in sorted(description_columns)]
    text = " ".join(filter(None, texts)).strip()
    return re.sub(r"\s+", " ", text)


def _to_float(value: Optional[str]) -> Optional[float]:
    """Best-effort conversion that also handles commas as decimal separators."""

    if value is None:
        return None
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return None
