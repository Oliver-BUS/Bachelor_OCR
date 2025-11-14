"""Cell cropping and OCR utilities used within the borehole pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from models import CellBBox

LOGGER = logging.getLogger(__name__)


def crop_cells(
    image_path: str,
    cell_bboxes: Sequence[CellBBox],
    output_dir: str,
    file_name: Optional[str] = None,
    page_index: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Crop every cell described in ``cell_bboxes`` and persist them to disk."""

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    base_name = Path(image_path).stem

    crops: List[Dict[str, object]] = []
    for bbox in cell_bboxes:
        # Slice the original image using the bounding-box coordinates.
        crop = image[bbox.y_min : bbox.y_max, bbox.x_min : bbox.x_max]
        cell_filename = (
            output_path / f"{base_name}_r{bbox.row_index:03d}_c{bbox.column_index:02d}.png"
        )
        cv2.imwrite(str(cell_filename), crop)

        crops.append(
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

    return crops


def ocr_cells(
    cells: Sequence[Dict[str, object]],
    engine: str = "tesseract",
    tesseract_config: str = "--psm 6",
    paddle_kwargs: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    """Run OCR per cropped cell using either Tesseract or PaddleOCR."""

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
    """Apply simple denoising/binarisation so OCR engines have cleaner input."""

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
    """Attach OCR text/metadata to the cell-level bookkeeping info."""

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
