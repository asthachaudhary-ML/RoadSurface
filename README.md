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
