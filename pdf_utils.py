"""PDF-to-image conversion helpers for the borehole OCR pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from pdf2image import convert_from_path

LOGGER = logging.getLogger(__name__)


def convert_pdf_to_images(input_pdf: str, output_dir: str, dpi: int = 300) -> List[str]:
    """Convert ``input_pdf`` into per-page PNGs saved under ``output_dir``."""

    # Ensure the output directory exists before writing files.
    pdf_path = Path(input_pdf)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Converting %s to images at %d dpi", pdf_path.name, dpi)
    pages = convert_from_path(str(pdf_path), dpi=dpi)

    image_paths: List[str] = []
    for idx, page in enumerate(pages):
        # Persist each page using a deterministic filename for later lookup.
        image_file = output_path / f"{pdf_path.stem}_page_{idx:03d}.png"
        page.save(image_file, "PNG")
        image_paths.append(str(image_file))

    return image_paths
