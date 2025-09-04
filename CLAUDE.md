# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ImageToSvg is a Python-based mathematical diagram vectorization tool that converts mathematical figures (geometry diagrams, charts) from raster images to high-quality SVG vector graphics. The project uses the Potrace vectorization engine for precise line extraction and mathematical shape recognition.

## Development Environment

### Python Environment
- Python 3.12+ required
- Uses pyproject.toml for dependency management
- Virtual environment located in `.venv/`

### Key Dependencies
- **Image Processing**: OpenCV, PIL/Pillow, scikit-image
- **OCR**: EasyOCR, PyTesseract, Tesseract
- **SVG Generation**: svgwrite
- **Automation**: autopy
- **Math/Array**: numpy

### External Dependencies
- **Potrace**: Must be installed system-wide and available in PATH
  - Used for professional-grade bitmap to vector conversion
  - Critical for the tool's core functionality

## Development Commands

### Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Running the Tool
```bash
# Run the main conversion tool
python potrace_image_to_svg.py

# Or execute directly (if executable)
./potrace_image_to_svg.py
```

### Testing
Check for potrace installation before running:
```bash
potrace --version
```

## Code Architecture

### Core Components

#### Main Converter (`potrace_image_to_svg.py`)
- **PotraceMathImageToSVG**: Main class handling the conversion pipeline
  - Image preprocessing with mathematical optimization
  - Potrace integration with specialized parameters
  - SVG post-processing and optimization

#### Key Methods
- `preprocess_image_for_potrace()`: Converts images to high-quality bitmaps using Otsu thresholding
- `run_potrace()`: Executes potrace with mathematical diagram-optimized parameters
- `batch_convert()`: Processes all images in input_image/ directory

#### Directory Structure
- `input_image/`: Source mathematical diagrams (PNG, JPG, BMP, TIFF)
- `output_svg/`: Generated SVG vector files
- `dist/`: Build artifacts
- `.venv/`: Python virtual environment

### Potrace Configuration
The tool uses specialized potrace parameters optimized for mathematical diagrams:
- `--tight`: Tight bounding boxes
- `--turnpolicy black`: Prefer black pixels for path decisions
- `--turdsize 2`: Minimum feature size in pixels
- `--alphamax 1.0`: Corner angle threshold
- `--opttolerance 0.2`: Optimization tolerance

## Future Architecture (Redesign Plan)

The codebase includes a redesign plan document (`精确数学配图SVG转换系统重设计方案.md`) outlining a 4-layer architecture:

1. **GeometryPrimitiveDetector**: High-level geometric shape detection
2. **ConstraintSolver**: Geometric constraint reasoning engine
3. **MathSemanticAnalyzer**: Mathematical semantics understanding
4. **PreciseSVGReconstructor**: Precise SVG generation with mathematical accuracy

## Working with Images

The tool expects mathematical diagrams containing:
- Geometric shapes (triangles, rectangles, circles)
- Mathematical annotations and labels
- Line drawings and technical diagrams

Input images are preprocessed with:
- Contrast enhancement
- Otsu thresholding for optimal binarization
- Format conversion to Potrace-compatible bitmaps

## Common Issues

- Ensure potrace is installed and in system PATH
- Input images should be clear with good contrast
- Mathematical diagrams work best with black lines on white background
- Large images may require timeout adjustments in subprocess calls