# Borehole Log OCR Pipeline

This repository contains a small, modular pipeline that converts borehole log PDFs
into structured CSV files (one row per depth interval).

The pipeline is designed for tables that contain at least:

- Textual description of the soil/rock layer
- Optional columns such as samples, remarks, etc.

## Overview of the steps

1. **PDF to images**  (`pdf_utils.convert_pdf_to_images`)
   - Each PDF page is rendered to a high-resolution PNG file.
   - Output is written under `<work_dir>/pages`.

2. **Table and cell detection** (`table_detection.detect_table_and_cells`)
   - Uses OpenCV (grayscale, thresholding, morphology) to detect the main table.
   - Extracts a grid of cell bounding boxes (row/column index + pixel coordinates).

3. **Cell cropping + OCR** (`cell_processing.crop_cells`, `cell_processing.ocr_cells`)
   - Crops each cell region to its own image under `<work_dir>/cells`.
   - Runs OCR per cell (Tesseract by default, PaddleOCR optional).
   - Produces one record per cell with:
     - file name, page index, row index, column index
     - bounding-box coordinates
     - OCR text and confidence

4. **Post-processing / CSV export** (`postprocess.postprocess_and_export`)
   - Groups cell records by `(file_name, page_index, row_index)`.
   - Parses depth intervals (e.g. `0.0-1.5`, `1.5 to 3.0`) from configured columns.
   - Concatenates non-depth columns into a single `description` field.
   - Writes a CSV with columns such as:
     - `file_name`, `page_index`, `row_index`
     - `depth_from_m`, `depth_to_m`
     - `description`
     - `raw_cells_json` (raw column text for debugging).

5. **Optional validation** (`postprocess.compare_with_manual`)
   - Compares the automatic CSV against a manually curated CSV.
   - Reports rows with mismatched depths or descriptions.

## Modules

- **`models.py`**
  - Defines shared dataclasses such as `CellBBox` and type aliases.

- **`pdf_utils.py`**
  - `convert_pdf_to_images(input_pdf, output_dir, dpi=300)`
  - Responsible only for rendering PDFs to image files.

- **`table_detection.py`**
  - `detect_table_and_cells(image_path, ...) -> list[CellBBox]`
  - Uses OpenCV morphology to detect the dominant table and derive the grid.

- **`cell_processing.py`**
  - `crop_cells(image_path, cell_bboxes, output_dir, ...)`
  - `ocr_cells(cells, engine="tesseract", ...)`
  - Handles cropping image cells and running OCR with either Tesseract or PaddleOCR.

- **`postprocess.py`**
  - `postprocess_and_export(records, output_csv_path, ...) -> pd.DataFrame`
  - `compare_with_manual(auto_csv_path, manual_csv_path, ...) -> pd.DataFrame`
  - Deals with depth parsing, description assembly, CSV export and validation.

- **`borehole_main.py`**
  - CLI/front-end that orchestrates all steps above via `process_pdf`.

## Running the pipeline

### 1. Install dependencies

Create/activate a virtual environment, then install the required packages, e.g.:

```bash
pip install opencv-python numpy pandas pdf2image pytesseract paddleocr
```

You also need:

- A local Tesseract installation (for `pytesseract`).
- `poppler` installed and on your PATH (for `pdf2image`, depending on OS).

### 2. Run from the command line

From the project directory:

```bash
python borehole_main.py path/to/input.pdf --work-dir outputs
# python borehole_main.py "C:\Admin\University\Bachelor Thesis\Datasets\PG-14373 - Logs-pages-1.pdf" --work-dir outputs
```

Useful flags:
- `--ocr-engine {tesseract,paddle}`: choose OCR backend per cell.
- `--dpi 300`: change rendering resolution for PDF -> images.
- `--output-csv some/path.csv`: choose explicit output file.
- `--depth-columns 0 1`: indices of depth-from / depth-to columns.
- `--description-columns 2 3`: indices that should be merged into the description.

If you omit `--output-csv`, a file named `<pdf_stem>_structured.csv` is created
under `<work_dir>`.


## Where to adjust for your data

- **Table detection parameters**: tweak `min_table_area_ratio` and
  `min_line_length_ratio` in `table_detection.detect_table_and_cells` if your
  tables are small, thick/thin lines, or have extra decorations.

- **Depth parsing**: adjust regexes in `postprocess.py` (`DEPTH_RANGE_PATTERN`,
  `NUMBER_PATTERN`) if your depth formats differ.

- **Description aggregation**: choose appropriate `--description-columns` or
  change `_build_description` if you want to keep more structure (e.g. separate
  remarks vs lithology).

This structure should make it easier to swap components (e.g. experiment with
different OCR settings or table-detection heuristics) without touching the rest
of the pipeline.
