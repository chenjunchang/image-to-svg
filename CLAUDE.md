# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ImageToSvg is a precision mathematical diagram SVG conversion system that transforms mathematical figures and geometry diagrams from raster images into mathematically accurate SVG vector graphics. The system uses a sophisticated 4-layer architecture with advanced computer vision, OCR, constraint solving, and semantic analysis to generate true geometric primitives instead of path-based approximations.

**Key Features:**
- Hybrid geometry detection using multiple algorithms (contour analysis, Hough transforms, template matching)
- Multi-engine OCR system with mathematical symbol recognition
- CAD-style geometric constraint solving
- Mathematical semantic analysis and scene understanding
- Precise SVG generation with true geometric primitives
- End-to-end conversion pipeline with confidence scoring

## Development Environment

### Python Environment
- Python 3.12+ required
- Uses pyproject.toml for dependency management
- Virtual environment located in `.venv/`

### Key Dependencies
- **Image Processing**: OpenCV, PIL/Pillow, scikit-image
- **OCR**: EasyOCR, PyTesseract, Tesseract
- **SVG Generation**: svgwrite
- **Mathematical Computing**: numpy, scipy
- **Graph Analysis**: networkx
- **Constraint Solving**: cvxpy, cvxopt (optional)
- **Testing**: pytest, unittest

### Optional Dependencies
- **GUI**: tkinter, PyQt5
- **Performance**: joblib, multiprocessing
- **Development**: black, flake8, mypy

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
# Run the main precision SVG converter
python precise_math_svg_converter.py input_image.png -o output.svg

# Run with intermediate result saving
python precise_math_svg_converter.py input_image.png -o output.svg --save-intermediates

# Run examples and demonstrations
python example_usage.py

# Batch conversion
python -c "from precise_math_svg_converter import PreciseMathSVGConverter; converter = PreciseMathSVGConverter(); converter.convert_batch(['img1.png', 'img2.png'], 'output_dir')"
```

### Testing
```bash
# Run comprehensive test suite
python test_framework.py -o test_results.json

# Run specific component tests
python -m pytest test_framework.py::ComponentTester -v

# Install additional dependencies for full testing
pip install -r requirements.txt
```

## Code Architecture

### 4-Layer Architecture System

The system implements a sophisticated 4-layer architecture for precise mathematical diagram conversion:

#### Layer 1: Detection and Recognition
- **HybridGeometryDetector** (`geometry_detector.py`): Multi-algorithm geometry detection
  - ContourBasedDetector: Contour analysis with Douglas-Peucker approximation
  - HoughBasedDetector: Line and circle detection using Hough transforms
  - TemplateMatchingDetector: Template-based shape recognition
  - Consensus fusion mechanism for high-accuracy detection

- **MathematicalOCR** (`mathematical_ocr.py`): Multi-engine text extraction
  - EasyOCR and Tesseract integration
  - Mathematical symbol recognition
  - Geometry-guided text extraction
  - Multi-engine result fusion

#### Layer 2: Constraint Solving
- **IncrementalConstraintSolver** (`constraint_solver.py`): CAD-style geometric constraints
  - ParallelConstraint, PerpendicularConstraint, IntersectionConstraint
  - Priority-based incremental solving
  - Numerical optimization for constraint satisfaction
  - Geometric relationship inference

#### Layer 3: Semantic Analysis
- **MathSemanticAnalyzer** (`math_semantic_analyzer.py`): Scene understanding
  - Mathematical pattern recognition
  - Scene graph construction using NetworkX
  - Geometry-text binding and relationship analysis
  - Context-aware mathematical interpretation

#### Layer 4: SVG Generation
- **PreciseSVGGenerator** (`precise_svg_generator.py`): Mathematical precision output
  - True geometric primitives (not path approximations)
  - Coordinate optimization and precision control
  - Layered SVG structure with validation
  - Mathematical metadata preservation

### Core Components

#### Main Converter (`precise_math_svg_converter.py`)
- **PreciseMathSVGConverter**: Main orchestrator class
  - `convert_image_to_svg()`: End-to-end conversion pipeline
  - `convert_batch()`: Batch processing with progress tracking
  - Intermediate result saving and confidence scoring
  - Comprehensive error handling and logging

#### Configuration System (`math_config.py`)
- **MathConfig**: Centralized configuration management
  - GeometryDetectionConfig: Detection algorithm parameters
  - OCRConfig: Multi-engine OCR settings
  - ConstraintSolverConfig: Constraint solving parameters
  - SVGGenerationConfig: Output precision and formatting

#### Geometric Primitives (`geometry_primitives.py`)
- **GeometryPrimitive**: Abstract base class for geometric elements
  - Point, Line, Circle, Triangle, Rectangle, Polygon classes
  - Precise distance calculations and geometric relationships
  - Bounding box computation and intersection testing

### File Structure

#### Core System Files
- `precise_math_svg_converter.py`: Main conversion pipeline
- `math_config.py`: Configuration system
- `geometry_primitives.py`: Geometric primitive definitions
- `geometry_detector.py`: Layer 1 - Geometry detection
- `mathematical_ocr.py`: Layer 1 - OCR and text recognition  
- `constraint_solver.py`: Layer 2 - Constraint solving
- `math_semantic_analyzer.py`: Layer 3 - Semantic analysis
- `precise_svg_generator.py`: Layer 4 - SVG generation

#### Testing and Validation
- `test_framework.py`: Comprehensive testing framework
- `example_usage.py`: Usage examples and demonstrations

#### Configuration and Documentation
- `requirements.txt`: Complete dependency specifications
- `pyproject.toml`: Package configuration
- `精确数学配图SVG转换系统重设计方案.md`: Design document (Chinese)

### Data Flow

1. **Image Input** → Image preprocessing and enhancement
2. **Layer 1** → Geometry detection + OCR text extraction  
3. **Layer 2** → Constraint solving and geometric refinement
4. **Layer 3** → Semantic analysis and scene understanding
5. **Layer 4** → Precise SVG generation with true primitives
6. **Output** → Mathematically accurate SVG with metadata

## Working with Images

### Supported Input Types
The system handles various mathematical diagrams:
- **Geometric figures**: Triangles, rectangles, circles, polygons, lines
- **Mathematical annotations**: Labels, equations, symbols, coordinates
- **Technical diagrams**: Engineering drawings, charts, graphs
- **Educational content**: Textbook diagrams, problem illustrations

### Image Preprocessing
- **Adaptive enhancement**: CLAHE histogram equalization
- **Noise reduction**: Gaussian filtering and morphological operations
- **Multi-format support**: PNG, JPG, BMP, TIFF input formats
- **Quality optimization**: Automatic contrast and brightness adjustment

### Output Quality
- **True geometric primitives**: Circles as `<circle>`, rectangles as `<rect>` (not path approximations)
- **Mathematical precision**: Configurable coordinate precision (default: 3 decimal places)
- **Semantic preservation**: Text-geometry relationships maintained
- **Scalable output**: Vector graphics with infinite zoom capability

## Usage Patterns

### Basic Conversion
```python
from precise_math_svg_converter import PreciseMathSVGConverter

converter = PreciseMathSVGConverter()
result = converter.convert_image_to_svg("diagram.png", "output.svg")
print(f"Success: {result.success}, Confidence: {result.confidence_score:.2%}")
```

### Advanced Usage with Configuration
```python
from math_config import MathConfig

config = MathConfig()
config.geometry_detection.algorithms = ['contour', 'hough', 'template']
config.ocr.engines = ['easyocr', 'tesseract']
config.svg_generation.coordinate_precision = 4

converter = PreciseMathSVGConverter(config)
```

## Common Issues and Solutions

### OCR Dependencies
- **EasyOCR installation**: May require CUDA for GPU acceleration
- **Tesseract setup**: Ensure tesseract is in system PATH
- **Fallback behavior**: System gracefully degrades if OCR engines unavailable

### Performance Optimization
- **Memory usage**: Large images may require chunked processing
- **Processing time**: Complex diagrams with many constraints take longer
- **Batch processing**: Use `convert_batch()` for multiple images

### Quality Issues
- **Low contrast images**: Use `config.preprocessing.apply_enhancement = True`
- **Complex diagrams**: Increase detection thresholds for better accuracy
- **Text recognition**: Ensure text is clear and not too small

### Troubleshooting
- **Check dependencies**: Run `python example_usage.py` to verify setup
- **Debug mode**: Use `save_intermediates=True` to inspect processing stages
- **Confidence scores**: Low scores (<0.5) indicate potential quality issues