# RoadSurface: Road Damage Detection and Severity Analysis Using Computer Vision

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Unit%20Tests-22%20Passing-brightgreen.svg)](tests/)

An automated, terminal-executable Computer Vision system designed to analyze visible road-surface distress (potholes, longitudinal/transverse cracks, surface voids) from optical imagery, quantify visual damage extent, and map geometric measurements into an interpretable structural severity rating.

---

## 1. Project Overview

Road surface deterioration presents serious hazards to road safety, vehicle longevity, and transportation infrastructure costs. Traditional road condition monitoring relies heavily on expensive specialized sensor vans equipped with laser profilometers, or manual road visual audits that are labor-intensive and subjective.

**RoadSurface** provides a lightweight, offline, classical Computer Vision pipeline capable of processing roadway imagery taken from dashcams, smartphones, or survey drones. Instead of treating damage inspection as a coarse binary classification or relying on black-box neural networks without spatial justification, RoadSurface detects defect boundaries, computes spatial pixel metrics, estimates surface degradation percentages, and calculates a multi-factor **Severity Score (0–100)** mapped into actionable engineering tiers (`LOW`, `MEDIUM`, `HIGH`).

---

## 2. Problem Statement

> **"Given an optical road image, estimate visible road-surface damage using classical computer vision techniques and classify the estimated severity using measurable image features rather than arbitrary labels."**

The primary contribution of this project is not merely detecting potholes, but converting raw image-space damage measurements into an objective, interpretable, and reproducible severity classification system.

---

## 3. Objectives

1. **Robust Preprocessing**: Normalize variable image dimensions while preserving aspect ratios, reduce asphalt sensor grain with Gaussian smoothing, and maintain feature integrity.
2. **Defect Candidate Extraction**: Segment dark cavity depressions and fissure edges using statistical modal luminance thresholding combined with adaptive contrast cues.
3. **Morphological Refinement**: Eliminate isolated gravel/aggregate artifacts and consolidate fragmented crack networks using mathematical morphology.
4. **Geometric Region Filtering**: Reject non-damage artifacts (small road aggregate, sky boundaries, high-aspect slivers) using area, spatial centroid, and bounding box criteria.
5. **Spatial Damage Quantification**: Accurately calculate total damaged pixel area, individual defect areas, and the percentage of visible road surface affected.
6. **Heuristic Severity Scoring**: Formulate a composite mathematical severity score $(0 - 100)$ incorporating surface percentage, defect frequency, and worst-case single defect magnitude.
7. **Visual and Structured Reporting**: Output an annotated image featuring bounding boxes, defect IDs, and a heads-up display (HUD) banner, along with a structured machine-readable JSON report.

---

## 4. Features

- **End-to-End Terminal Execution**: Zero reliance on GUI tools or interactive notebooks; runs directly from PowerShell, Bash, or Command Prompt.
- **Pure Classical Computer Vision**: Relies on OpenCV and NumPy; requires no GPU, CUDA drivers, or heavy deep-learning frameworks.
- **Aspect-Ratio Preserving Scaling**: Handles arbitrary camera resolutions without image distortion.
- **Statistical Pavement Modeling**: Automatically estimates asphalt background distribution to prevent false positives on clean pavement.
- **Configurable CLI Parameters**: Customizable threshold modes, minimum area filters, and intermediate stage exports.
- **Interpretable Heuristic Classification**: Transparent mathematical mapping from pixel counts to `LOW`, `MEDIUM`, and `HIGH` severity.
- **Visual Heads-Up Display (HUD)**: Color-coded severity indicators, semi-transparent region fills, and defect dimension tags.
- **Detailed JSON Export**: Granular defect coordinates $(x, y, w, h)$, centroids, and individual areas for downstream GIS or municipal database integration.
- **Self-Contained Automated Unit Tests**: Comprehensive 22-test suite testing edge cases, boundary conditions, and corrupt inputs.

---

## 5. Computer Vision Pipeline

```text
                       Input Image (JPG / PNG)
                                  │
                                  ▼
                     +──────────────────────────+
                     │    Input Validation      │  (Verify existence, format, readability)
                     +──────────────────────────+
                                  │
                                  ▼
                     +──────────────────────────+
                     │   Image Preprocessing    │  (Aspect-ratio resize, Grayscale conversion,
                     │                          │   Gaussian blur noise suppression)
                     +──────────────────────────+
                                  │
                                  ▼
                     +──────────────────────────+
                     │   Candidate Extraction   │  (Statistical modal luminance threshold
                     │                          │   fused with adaptive gradient cues)
                     +──────────────────────────+
                                  │
                                  ▼
                     +──────────────────────────+
                     │ Morphological Processing │  (MORPH_OPEN to clear gravel noise,
                     │                          │   MORPH_CLOSE to bridge crack fissures)
                     +──────────────────────────+
                                  │
                                  ▼
                     +──────────────────────────+
                     │    Contour Detection     │  (cv2.findContours external boundaries)
                     +──────────────────────────+
                                  │
                                  ▼
                     +──────────────────────────+
                     │     Region Filtering     │  (Filter by min_area, max_area, aspect
                     │                          │   ratio, and upper-horizon exclusion)
                     +──────────────────────────+
                                  │
                                  ▼
                     +──────────────────────────+
                     │   Damage Quantification  │  (Sum pixel areas, calculate surface %
                     │    & Severity Scoring    │   and compute 0-100 severity index)
                     +──────────────────────────+
                                  │
                                  ▼
                     +──────────────────────────+
                     │    Visualization & I/O   │  (Render HUD banner, bounding boxes,
                     │                          │   save annotated image & analysis.json)
                     +──────────────────────────+
```

---

## 6. Technologies Used

- **Language**: Python 3.9+ (Tested and verified on Python 3.13)
- **Computer Vision**: OpenCV (`opencv-python >= 4.8.0`)
- **Numerical Processing**: NumPy (`numpy >= 1.24.0`)
- **Image Operations**: Pillow (`pillow >= 10.0.0`)
- **Standard Libraries**: `argparse`, `pathlib`, `json`, `time`, `csv`, `unittest`

---

## 7. Project Structure

```text
RoadSurface/
│
├── data/
│   ├── input/
│   │   ├── sample_road.jpg         # Test image with moderate potholes and cracks
│   │   ├── clean_road.jpg          # Pristine road image (negative control)
│   │   └── severe_road.jpg         # Severely distressed pavement image
│   │
│   └── output/
│       ├── annotated_road.jpg      # Output image with bounding boxes & HUD
│       ├── analysis.json           # Machine-readable JSON summary report
│       └── intermediate/           # Saved stage images (--save-intermediate)
│           ├── blurred.jpg
│           ├── contours.jpg
│           ├── edges.jpg
│           ├── enhanced.jpg
│           ├── grayscale.jpg
│           ├── morphology.jpg
│           └── threshold.jpg
│
├── src/
│   ├── __init__.py                 # Package declaration and version metadata
│   ├── main.py                     # CLI entry point and pipeline orchestrator
│   ├── preprocessing.py            # Resizing, color conversion, blurring, Canny
│   ├── damage_detection.py         # Statistical/adaptive thresholding & contour filtering
│   ├── severity.py                 # Mathematical metrics, area calculations & severity index
│   ├── visualization.py            # HUD rendering, bounding boxes, and intermediate export
│   └── utils.py                    # Input validation, terminal banner, and JSON output
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py       # Unit tests for preprocessing operations
│   ├── test_detection.py           # Unit tests for candidate extraction and filtering
│   ├── test_severity.py            # Unit tests for metric math and scoring boundaries
│   └── test_utils.py               # Unit tests for validation and file I/O
│
├── create_samples.py               # Procedural generator for synthetic benchmark images
├── experiments.py                  # Batch evaluation script generating results.csv
├── requirements.txt                # Pinned pip dependencies
├── results.csv                     # Measured benchmark metrics and execution times
├── REPORT.md                       # Comprehensive university academic project report
├── .gitignore                      # Python and IDE exclusions
├── LICENSE                         # MIT License
└── README.md                       # Complete project documentation
```

---

## 8. Installation

### Step 1: Clone or Open the Repository
```bash
cd RoadSurface
```

### Step 2: Create a Virtual Environment (Recommended)

On **Windows (PowerShell)**:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On **Linux / macOS**:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 9. Running the Project

Run the application on the sample road image:

```bash
python src/main.py --input data/input/sample_road.jpg
```

To run on your own image:
```bash
python src/main.py --input path/to/your_road_image.jpg
```

To enable intermediate stage outputs:
```bash
python src/main.py --input data/input/sample_road.jpg --save-intermediate
```

---

## 10. Optional Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input`, `-i` | `str` | `data/input/sample_road.jpg` | Path to the target road image (`.jpg`, `.jpeg`, `.png`) |
| `--output`, `-o` | `str` | `data/output` | Directory where output images and reports are saved |
| `--threshold`, `-t` | `str` | `auto` | Candidate extraction mode: `auto`, `otsu`, `adaptive`, or an integer (e.g. `80`) |
| `--min-area` | `int` | `150` | Minimum contour area in pixels to eliminate fine gravel noise |
| `--clahe` | `flag` | `False` | Enables CLAHE contrast enhancement (disabled by default to prevent noise amplification) |
| `--max-dimension` | `int` | `1280` | Maximum width/height for resizing while preserving aspect ratio |
| `--save-intermediate` | `flag` | `False` | Saves transformation images to `data/output/intermediate/` |

---

## 11. Example Output

Executing the standard command:
```bash
python src/main.py --input data/input/sample_road.jpg
```

Yields the following verified terminal output:

```text
=========================================
        ROADSURFACE ANALYZER             
=========================================
Input image        : sample_road.jpg
Image resolution   : 1280 x 720

Preprocessing      : Completed
Damage extraction  : Completed
Contour analysis   : Completed

-----------------------------------------
RESULTS
-----------------------------------------
Damage regions     : 5
Damaged area       : 2.69%
Largest region     : 1.35%
Severity score     : 30.2 / 100
Severity level     : MEDIUM
-----------------------------------------

Annotated image:
data\output\annotated_road.jpg

Analysis report:
data\output\analysis.json
=========================================
```

When evaluated on pristine control pavement (`data/input/clean_road.jpg`):
```text
Damage regions     : 0
Damaged area       : 0.00%
Largest region     : 0.00%
Severity score     : 0.0 / 100
Severity level     : LOW
```

When evaluated on severe hazard pavement (`data/input/severe_road.jpg`):
```text
Damage regions     : 9
Damaged area       : 7.21%
Largest region     : 3.39%
Severity score     : 63.2 / 100
Severity level     : HIGH
```

---

## 12. Output Files

Each execution produces two primary artifacts in the specified output directory:

### 1. `annotated_road.jpg`
A high-resolution visualization displaying:
- **Contour Infill**: Translucent color-coded shading highlighting exact defect geometry.
- **Bounding Boxes**: Rectangular boundaries demarcating each candidate region.
- **Region Badges**: Individual identifiers `#1`, `#2`, ... with calculated pixel areas.
- **Heads-Up Display (HUD)**: A semi-transparent information card on the top-right detailing total detected regions, damaged area percentage, numeric severity score, and categorical severity rating.

### 2. `analysis.json`
A machine-readable JSON document structuring complete quantitative findings:
```json
{
    "input_image": "sample_road.jpg",
    "image_width": 1280,
    "image_height": 720,
    "processed_width": 1280,
    "processed_height": 720,
    "damage_regions": 5,
    "damage_percentage": 2.69,
    "severity_score": 30.2,
    "severity_level": "MEDIUM",
    "metrics": {
        "road_area_pixels": 921600,
        "damage_area_pixels": 24799,
        "largest_region_pixels": 12404,
        "largest_region_percentage": 1.35
    },
    "regions": [
        {
            "id": 1,
            "area_pixels": 12404,
            "x": 391,
            "y": 429,
            "width": 180,
            "height": 106,
            "aspect_ratio": 1.7,
            "extent": 0.65,
            "centroid": {
                "x": 480,
                "y": 479
            }
        }
    ]
}
```

---

## 13. Methodology

### A. Preprocessing and Noise Suppression
Raw roadway images often contain high-frequency sensor noise and gravel texture.
1. **Aspect-Ratio Resizing**: Large images are downscaled so that $\max(W, H) \le 1280$, reducing computational overhead while avoiding aspect ratio distortion.
2. **Grayscale Conversion**: Eliminates color variance since asphalt distress is primarily distinguished by luminance contrast and edge transitions.
3. **Gaussian Smoothing**: An odd-sized Gaussian kernel ($5 \times 5$, $\sigma = 1.2$) suppresses high-frequency aggregate noise while preserving prominent distress perimeters.
4. **Contrast Considerations**: Contrast Limited Adaptive Histogram Equalization (CLAHE) was empirically evaluated. While CLAHE enhances local contrast, experiments demonstrated that applying CLAHE blindly to uniform asphalt stretches background noise variance, occasionally inducing false positives on clean pavement. Consequently, CLAHE is disabled by default and accessible via `--clahe`.

### B. Statistical Pavement Modeling & Candidate Extraction
Standard Otsu thresholding assumes a bimodal distribution with two balanced peaks. On road surfaces, normal pavement accounts for $>95\%$ of pixels, causing global Otsu to split the unimodal noise distribution in half.

To solve this, **RoadSurface** models the pavement's modal background:
$$\text{Threshold}_{\text{stat}} = \max\left(25, \, \text{Median}(I) - 2.0 \cdot \sigma(I)\right)$$
Pixels darker than this threshold represent cavity depressions. Concurrently, an adaptive Gaussian threshold ($B = 31, C = 16$) captures high-contrast fissure crack transitions. The union of these two masks yields a robust candidate segmentation.

### C. Morphological Refinement
A $5 \times 5$ elliptical structuring element is applied:
- **Morphological Opening** ($\text{Erosion} \to \text{Dilation}$): Eradicates isolated aggregate pixels and sensor grain.
- **Morphological Closing** ($\text{Dilation} \to \text{Erosion}$): Seals microscopic gaps within pothole cavities and links fractured crack trajectories.

### D. Contour Detection & Geometric Filtering
External contours are extracted via `cv2.findContours(..., cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`. Each candidate contour undergoes heuristic filtering:
1. **Minimum Area**: Discards regions with $\text{Area} < 150 \text{ px}$ (gravel, cigarette butts, oil spots).
2. **Maximum Area**: Rejects contours exceeding $35\%$ of the total frame (large vehicle shadows, overpass occlusions).
3. **Top Margin Exclusion**: Rejects detections in the top $5\%$ of the image, which in forward-facing cameras typically correspond to horizon, sky, or tree lines.
4. **Aspect Ratio Sanity Check**: Discards degenerate single-pixel boundary slivers where $W/H < 0.05$ or $W/H > 20.0$.

---

## 14. Severity Calculation

The severity calculation converts image-space physical measurements into an interpretable rating between $0$ and $100$.

> [!NOTE]
> All area measurements are computed in **image-space pixels**. Without camera extrinsic calibration or a physical reference fiducial, image measurements do not directly equate to square meters.

### Heuristic Mathematical Formulation

The composite Severity Score combines three complementary indicators:
1. **Overall Damage Extent** ($S_{\text{area}}$): Total percentage of the visible roadway affected, normalized against a severe threshold of $15.0\%$:
   $$S_{\text{area}} = \min\left(100.0, \, \frac{P_{\text{damage}}}{15.0} \times 100.0\right)$$
2. **Defect Frequency / Dispersion** ($S_{\text{count}}$): Number of distinct detected damage clusters, normalized against a critical threshold of $8$ defect clusters:
   $$S_{\text{count}} = \min\left(100.0, \, \frac{N_{\text{regions}}}{8.0} \times 100.0\right)$$
3. **Acute Single-Defect Hazard** ($S_{\text{worst}}$): Ratio of the largest single crater/pothole to road area, normalized against an acute threshold of $6.0\%$:
   $$S_{\text{worst}} = \min\left(100.0, \, \frac{P_{\text{largest}}}{6.0} \times 100.0\right)$$

The composite score is computed using a weighted linear combination:
$$\text{Score} = \min\left(100.0, \, \left(0.50 \cdot S_{\text{area}}\right) + \left(0.25 \cdot S_{\text{count}}\right) + \left(0.25 \cdot S_{\text{worst}}\right)\right)$$

### Severity Level Classification

| Score Range | Severity Level | Practical Interpretation |
| :---: | :---: | :--- |
| **0.0 – 30.0** | **LOW** | Minor surface defects; superficial hairline cracks; no immediate vehicle risk. |
| **30.1 – 60.0** | **MEDIUM** | Noticeable potholes or crack clusters; requires routine maintenance monitoring. |
| **60.1 – 100.0** | **HIGH** | Severe structural cavity depressions; hazardous to vehicle suspension and tires. |

---

## 15. Limitations

1. **Illumination and Shadowing**: Heavy tree canopy shadows or low-angle sunlight can create localized dark patterns that classical thresholding may interpret as candidate depressions.
2. **Image-Space Perspective Distortion**: In forward-looking vehicle cameras, road pixels near the horizon represent vastly more ground area than pixels in the near foreground. A perspective bird's-eye transform (inverse perspective mapping) would be required for metric square-meter quantification.
3. **Absence of 3D Volumetric Depth**: Monocular 2D images capture surface reflectivity, not depth. A dark stain (e.g. spilled motor oil or fresh asphalt patch) might have low luminance similar to a shallow pothole.
4. **Road Surface Variance**: Pavements constructed with light-colored concrete vs. dark asphalt exhibit differing contrast dynamics requiring adaptive parameter tuning.

---

## 16. Future Scope

1. **Stereo Vision / Depth Integration**: Incorporate dual-camera disparity estimation or LiDAR point clouds to measure true crater depth and volumetric displacement.
2. **Deep-Learning Semantic Segmentation**: Integrate lightweight segmentation architectures (e.g., MobileNet-UNet) for pixel-precise crack delineation under challenging lighting.
3. **Inverse Perspective Mapping (IPM)**: Calibrate camera extrinsics to project perspective views onto an orthogonal top-down grid for true metric distance measurement.
4. **Temporal Video Tracking**: Extend the pipeline from single images to continuous video feeds, applying Kalman filtering to track defects across consecutive frames and prevent double-counting.
5. **GPS & Municipal Cloud Sync**: Tag detected damage with onboard GPS coordinates and automatically upload road distress reports to municipal maintenance dashboards.

---

## 17. Testing

The project includes a comprehensive, automated test suite utilizing Python's native `unittest` framework:

To run all unit tests:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### Verified Test Suite Breakdown (22 Tests Passing)
- `tests/test_preprocessing.py`:
  - `test_resize_maintaining_aspect_ratio_large`: Verifies aspect ratio preservation when downscaling.
  - `test_resize_maintaining_aspect_ratio_small`: Confirms images below maximum dimension remain untouched.
  - `test_to_grayscale`: Validates conversion to single-channel 8-bit array and idempotency.
  - `test_apply_noise_reduction`: Verifies suppression of impulse noise through Gaussian filtering.
  - `test_enhance_contrast`: Verifies CLAHE output dimensions and type consistency.
  - `test_detect_edges`: Checks that Canny produces strict binary maps with values in $\{0, 255\}$.
  - `test_run_preprocessing_pipeline`: Validates the complete preprocessing dictionary contract.
- `tests/test_detection.py`:
  - `test_extract_candidate_mask_modes`: Tests `auto`, `otsu`, and `adaptive` modes.
  - `test_extract_candidate_mask_manual`: Verifies manual threshold cutoff behavior.
  - `test_apply_morphological_refinement`: Asserts isolated single-pixel noise is eradicated by opening while primary contours persist.
  - `test_detect_candidate_contours`: Validates extraction of geometric contours.
  - `test_filter_damage_regions`: Asserts tiny noise particles and upper horizon artifacts are filtered out.
- `tests/test_severity.py`:
  - `test_damage_metrics_calculation`: Validates damaged pixel sum, percentages, and largest region calculations.
  - `test_damage_metrics_empty_regions`: Verifies zero-damage boundary handling.
  - `test_severity_score_bounds_and_zero`: Verifies mathematical scoring bounds $[0.0, 100.0]$.
  - `test_classify_severity_level`: Asserts exact classification into `LOW`, `MEDIUM`, and `HIGH`.
  - `test_analyze_severity_orchestrator`: Tests high-level dictionary structure.
- `tests/test_utils.py`:
  - `test_validate_image_path_success`: Verifies path resolution for existing images.
  - `test_validate_image_path_not_found`: Confirms `FileNotFoundError` for missing files.
  - `test_validate_image_path_unsupported_extension`: Confirms rejection of non-image extensions.
  - `test_validate_image_path_corrupted`: Confirms graceful handling of non-decodable files.
  - `test_save_json_report`: Verifies formatted JSON serialization and disk persistence.

---

## 18. Ethical and Practical Considerations

- **Experimental Status**: This software is an educational and prototype engineering tool. It is **not** certified as an official civil infrastructure inspection standard.
- **Safety Criticality**: Autonomous road repair or lane-closure decisions must involve certified structural engineers and ground-truth physical verification.
- **Privacy Assurance**: The system processes roadway surface textures. If deployed in real-time camera setups, faces and vehicle license plates should be blurred prior to archiving to safeguard privacy.

---

## 19. License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for full details.
