from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from pdf2image import convert_from_path


LOGGER = logging.getLogger(__name__)

RangeMatch = Tuple[Optional[float], Optional[float]]


@dataclass(frozen=True)
class CellBBox:
    row_index: int
    column_index: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int


DEPTH_RANGE_PATTERN = re.compile(
    r"(?P<from>\d+(?:[\.,]\d+)?)\s*(?:-|to|–|—)\s*(?P<to>\d+(?:[\.,]\d+)?)",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"\d+(?:[\.,]\d+)?")


def convert_pdf_to_images(input_pdf: str, output_dir: str, dpi: int = 300) -> List[str]:
    """Convert a PDF into page-level PNG images and return their paths."""

    pdf_path = Path(input_pdf)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Converting %s to images at %d dpi", pdf_path, dpi)
    pages = convert_from_path(str(pdf_path), dpi=dpi)

    image_paths: List[str] = []
    for idx, page in enumerate(pages):
        image_file = output_path / f"{pdf_path.stem}_page_{idx:03d}.png"
        page.save(image_file, "PNG")
        image_paths.append(str(image_file))

    return image_paths


def detect_table_and_cells(
    image_path: str,
    min_table_area_ratio: float = 0.2,
    min_line_length_ratio: float = 0.3,
) -> List[CellBBox]:
    """Detect the dominant table within an image and return its cells as bounding boxes."""

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

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
        for col_idx in range(len(col_boundaries) - 1):
            y_min = row_boundaries[row_idx]
            y_max = row_boundaries[row_idx + 1]
            x_min = col_boundaries[col_idx]
            x_max = col_boundaries[col_idx + 1]
            if (x_max - x_min) < 5 or (y_max - y_min) < 5:
                continue
            cells.append(
                CellBBox(
                    row_index=row_idx,
                    column_index=col_idx,
                    x_min=int(x_min),
                    y_min=int(y_min),
                    x_max=int(x_max),
                    y_max=int(y_max),
                )
            )

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


def _grid_boundaries(
    line_image: np.ndarray,
    axis: int,
    offset: int,
    min_line_length_ratio: float,
    fallback_start: int,
    fallback_end: int,
) -> List[int]:
    projection = np.sum(line_image > 0, axis=axis)
    max_val = projection.max()
    if max_val == 0:
        return [fallback_start, fallback_end]

    threshold = max_val * min_line_length_ratio
    segments: List[Tuple[int, int]] = []
    start_idx: Optional[int] = None
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

    deduped: List[int] = []
    for value in boundaries:
        if not deduped or value - deduped[-1] > 2:
            deduped.append(value)
        else:
            deduped[-1] = value

    return deduped


def crop_cells(
    image_path: str,
    cell_bboxes: Sequence[CellBBox],
    output_dir: str,
    file_name: Optional[str] = None,
    page_index: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Crop detected cell regions and persist them to disk."""

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_name = Path(image_path).stem
    meta: List[Dict[str, object]] = []
    for bbox in cell_bboxes:
        crop = image[bbox.y_min : bbox.y_max, bbox.x_min : bbox.x_max]
        cell_filename = (
            output_path
            / f"{base_name}_r{bbox.row_index:03d}_c{bbox.column_index:02d}.png"
        )
        cv2.imwrite(str(cell_filename), crop)

        meta.append(
            {
                "cell_image_path": str(cell_filename),
                "row_index": bbox.row_index,
                "column_index": bbox.column_index,
                "bbox": {
                    "x_min": bbox.x_min,
                    "y_min": bbox.y_min,
                    "x_max": bbox.x_max,
                    "y_max": bbox.y_max,
                },
                "file_name": file_name or Path(image_path).name,
                "page_index": page_index if page_index is not None else 0,
                "page_image_path": image_path,
            }
        )

    return meta


def ocr_cells(
    cells: Sequence[Dict[str, object]],
    engine: str = "tesseract",
    tesseract_config: str = "--psm 6",
    paddle_kwargs: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    """Run OCR on each cropped cell image and attach metadata."""

    records: List[Dict[str, object]] = []

    if engine.lower() == "paddle":
        from paddleocr import PaddleOCR  # type: ignore

        ocr = PaddleOCR(**(paddle_kwargs or {"use_angle_cls": True, "lang": "en"}))
        for cell in cells:
            result = ocr.ocr(cell["cell_image_path"], cls=True)
            texts: List[str] = []
            scores: List[float] = []
            for line in result:
                for _, (text, score) in line:
                    texts.append(text)
                    scores.append(float(score))
            records.append(
                _build_record(
                    cell,
                    text=" ".join(texts).strip(),
                    confidence=float(np.mean(scores)) if scores else None,
                    engine="paddle",
                )
            )
        return records

    import pytesseract
    from PIL import Image

    for cell in cells:
        prepared = _prepare_for_ocr(cell["cell_image_path"])
        pil_image = Image.fromarray(prepared)
        data = pytesseract.image_to_data(
            pil_image, output_type=pytesseract.Output.DICT, config=tesseract_config
        )
        words = [word.strip() for word in data["text"] if word.strip()]
        confidences = [float(conf) for conf in data["conf"] if conf not in {"-1", "-0"}]
        text = " ".join(words)
        records.append(
            _build_record(
                cell,
                text=text,
                confidence=float(np.mean(confidences)) if confidences else None,
                engine="tesseract",
            )
        )

    return records


def _prepare_for_ocr(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image for OCR: {image_path}")
    image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _build_record(
    cell: Dict[str, object],
    text: str,
    confidence: Optional[float],
    engine: str,
) -> Dict[str, object]:
    bbox = cell["bbox"]
    return {
        "file_name": cell["file_name"],
        "page_index": cell["page_index"],
        "row_index": cell["row_index"],
        "column_index": cell["column_index"],
        "x_min": bbox["x_min"],
        "y_min": bbox["y_min"],
        "x_max": bbox["x_max"],
        "y_max": bbox["y_max"],
        "text": text.strip(),
        "confidence": confidence,
        "ocr_engine": engine,
        "cell_image_path": cell["cell_image_path"],
        "page_image_path": cell["page_image_path"],
    }


def postprocess_and_export(
    records: Sequence[Dict[str, object]],
    output_csv_path: str,
    depth_columns: Sequence[int] = (0, 1),
    description_columns: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Aggregate cell OCR into depth intervals and export to CSV."""

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

    grouped = []
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


def _extract_depths(
    column_text: Dict[int, str],
    depth_columns: Sequence[int],
) -> RangeMatch:
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
    if description_columns is None:
        description_columns = [col for col in column_text if col not in depth_columns]

    texts = [column_text.get(col, "") for col in sorted(description_columns)]
    text = " ".join(filter(None, texts)).strip()
    return re.sub(r"\s+", " ", text)


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return None


def compare_with_manual(
    auto_csv_path: str,
    manual_csv_path: str,
    id_columns: Sequence[str] = ("file_name", "page_index", "row_index"),
    depth_tolerance: float = 0.05,
) -> pd.DataFrame:
    """Compare generated CSV with a manual reference and highlight differences."""

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


def process_pdf(
    input_pdf: str,
    work_dir: str,
    ocr_engine: str = "tesseract",
    dpi: int = 300,
    depth_columns: Sequence[int] = (0, 1),
    description_columns: Optional[Sequence[int]] = None,
    output_csv_path: Optional[str] = None,
    paddle_kwargs: Optional[Dict[str, object]] = None,
    tesseract_config: str = "--psm 6",
) -> pd.DataFrame:
    """End-to-end helper that runs the entire pipeline for a PDF."""

    work_path = Path(work_dir)
    images_dir = work_path / "pages"
    cells_dir = work_path / "cells"
    output_csv = output_csv_path or work_path / f"{Path(input_pdf).stem}_structured.csv"

    images = convert_pdf_to_images(input_pdf, str(images_dir), dpi=dpi)
    all_records: List[Dict[str, object]] = []

    for page_idx, image_path in enumerate(images):
        try:
            cell_bboxes = detect_table_and_cells(image_path)
        except ValueError as exc:
            LOGGER.warning("Page %d: %s", page_idx, exc)
            continue

        crops = crop_cells(
            image_path,
            cell_bboxes,
            output_dir=str(cells_dir),
            file_name=Path(input_pdf).name,
            page_index=page_idx,
        )
        records = ocr_cells(
            crops,
            engine=ocr_engine,
            paddle_kwargs=paddle_kwargs,
            tesseract_config=tesseract_config,
        )
        all_records.extend(records)

    if not all_records:
        LOGGER.warning("No OCR records produced for %s", input_pdf)

    return postprocess_and_export(
        all_records,
        output_csv_path=str(output_csv),
        depth_columns=depth_columns,
        description_columns=description_columns,
    )


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Borehole table OCR pipeline")
    parser.add_argument("input_pdf", help="Path to the borehole log PDF")
    parser.add_argument(
        "--work-dir",
        default="pipeline_outputs",
        help="Directory for intermediate images and crops",
    )
    parser.add_argument(
        "--ocr-engine",
        choices=["tesseract", "paddle"],
        default="tesseract",
        help="OCR engine to use per cell",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="DPI for the PDF to image conversion"
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path for the structured CSV output",
    )
    parser.add_argument(
        "--depth-columns",
        type=int,
        nargs="*",
        default=(0, 1),
        help="Column indices that contain depth-from / depth-to information",
    )
    parser.add_argument(
        "--description-columns",
        type=int,
        nargs="*",
        default=None,
        help="Column indices to merge into the description field",
    )
    parser.add_argument(
        "--tesseract-config",
        default="--psm 6",
        help="Extra configuration passed to pytesseract",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_cli()
    args = parser.parse_args()

    process_pdf(
        input_pdf=args.input_pdf,
        work_dir=args.work_dir,
        ocr_engine=args.ocr_engine,
        dpi=args.dpi,
        depth_columns=tuple(args.depth_columns),
        description_columns=tuple(args.description_columns)
        if args.description_columns is not None
        else None,
        output_csv_path=args.output_csv,
        tesseract_config=args.tesseract_config,
    )


if __name__ == "__main__":
    main()
