# Repository Guidelines

## Project Structure & Module Organization
- `*.py`: Core modules (geometry detection, constraints, OCR, SVG generation).
- `example_usage.py`, `quick_test.py`, `test_geometry_only.py`: Runbook-style examples/tests.
- `validate_conversion.py`: Validates SVG output quality and metrics.
- `input_image/`: Sample or user-provided input bitmaps.
- `output_svg/`, `output_svg_validation/`: Generated SVGs and validation artifacts.
- `dist/`, `imagetosvg.egg-info/`: Build artifacts (do not edit).

## Build, Test, and Development Commands
- Run basic demo: `python example_usage.py` (generates test images and SVGs).
- Quick geometry test: `python quick_test.py`.
- Geometry-only flow: `python test_geometry_only.py`.
- Validate outputs: `python validate_conversion.py input_image/ sample.png`.
- Batch conversion (API): instantiate `PreciseMathSVGConverter` and call `convert_batch(...)`.

## Coding Style & Naming Conventions
- Python 3; follow PEP 8; indent with 4 spaces.
- Classes: `PascalCase` (e.g., `HybridGeometryDetector`); functions/variables: `snake_case`.
- Use type hints and concise docstrings; prefer `@dataclass` for simple data containers.
- Keep modules single‑purpose: primitives → detection → OCR → constraints → SVG.

## Testing Guidelines
- Prefer script-based tests in `test_*.py` alongside modules.
- Add small, deterministic images to `input_image/` for new cases.
- Name tests descriptively (e.g., `test_arc_detection_case1.py`).
- Run locally with the commands above; include before/after artifacts in `output_svg/`.

## Commit & Pull Request Guidelines
- Commits: small, focused, imperative mood (e.g., "Add circle fit tolerance").
- PRs: include purpose, summary of changes, and screenshots/links to SVGs in `output_svg/`.
- Reference issues (e.g., `Fixes #12`) and list manual test steps.
- Avoid reformat‑only diffs; keep build artifacts out of PRs.

## Security & Configuration Tips
- OCR backends: EasyOCR (GPU optional) or Tesseract; ensure binaries/weights available.
- External deps: OpenCV, NumPy, SciPy, NetworkX, EasyOCR/PyTorch or `pytesseract`.
- Large models and outputs should not be committed; use `.gitignore` patterns already present.

