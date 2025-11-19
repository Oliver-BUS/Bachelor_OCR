"""CLI entry point that orchestrates the modular borehole log OCR pipeline.

This script wires together the individual steps implemented in:
- pdf_utils.convert_pdf_to_images: PDF -> per-page images
- table_detection.detect_table_and_cells: find table grid + cell boxes
- cell_processing.crop_cells / ocr_cells: crop images + OCR per cell
- postprocess.postprocess_and_export: aggregate OCR into structured CSV
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from cell_processing import crop_cells, ocr_cells
from debug_vis import draw_cells_overlay
from pdf_utils import convert_pdf_to_images
from postprocess import postprocess_and_export, export_cells_flat
from table_detection import detect_table_and_cells

LOGGER = logging.getLogger(__name__)


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
) -> Path:
    """Run the full PDF -> CSV pipeline and return the structured CSV path."""

    # Prepare working folders that hold intermediate outputs.
    work_path = Path(work_dir)
    images_dir = work_path / "pages"
    cells_dir = work_path / "cells"
    debug_dir = work_path / "annotated"
    work_path.mkdir(parents=True, exist_ok=True)

    output_csv = (
        Path(output_csv_path)
        if output_csv_path is not None
        else work_path / f"{Path(input_pdf).stem}_structured.csv"
    )

    LOGGER.info("Starting processing for %s", input_pdf)

    # 1) Convert PDF pages into high-resolution PNG images.
    images = convert_pdf_to_images(input_pdf, str(images_dir), dpi=dpi)

    all_records: List[Dict[str, object]] = []
    for page_idx, image_path in enumerate(images):
        LOGGER.info("Page %d: detecting table structure", page_idx)

        try:
            cell_bboxes = detect_table_and_cells(image_path)
        except ValueError as exc:
            LOGGER.warning("Page %d skipped: %s", page_idx, exc)
            continue

        # 2) Crop every detected cell so OCR runs on cell-level images.
        crops = crop_cells(
            image_path,
            cell_bboxes,
            output_dir=str(cells_dir),
            file_name=Path(input_pdf).name,
            page_index=page_idx,
        )

        # 3) Run OCR on each cell image (Tesseract by default, Paddle optional).
        page_records = ocr_cells(
            crops,
            engine=ocr_engine,
            paddle_kwargs=paddle_kwargs,
            tesseract_config=tesseract_config,
        )
        all_records.extend(page_records)

        # 4) Write an annotated debug image showing all cells (green) and
        #    those with non-empty OCR text (inner red box).
        debug_image_path = debug_dir / f"{Path(image_path).stem}_annotated.png"
        draw_cells_overlay(image_path, cell_bboxes, str(debug_image_path), text_records=page_records)

    if not all_records:
        LOGGER.warning("No OCR records produced for %s", input_pdf)

    # 4) Aggregate cell OCR into depth intervals and export to CSV.
    LOGGER.info("Post-processing %d cell records", len(all_records))
    postprocess_and_export(
        all_records,
        output_csv_path=str(output_csv),
        depth_columns=depth_columns,
        description_columns=description_columns,
    )

    # Additionally, export a flat per-cell CSV for simpler inspection.
    cells_csv = work_path / f"{Path(input_pdf).stem}_cells.csv"
    export_cells_flat(all_records, output_csv_path=str(cells_csv))

    LOGGER.info("Pipeline finished -> %s", output_csv)
    return output_csv


def _build_cli() -> argparse.ArgumentParser:
    """Create an argument parser so this module can be invoked as a script."""

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
        "--dpi",
        type=int,
        default=300,
        help="DPI for the PDF to image conversion",
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
        help="Column indices that contain depth-from/to information",
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
    """Script entry point when running ``python borehole_main.py``."""

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
