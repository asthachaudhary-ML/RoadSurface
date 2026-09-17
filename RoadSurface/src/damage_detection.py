"""
Damage Detection and Contour Analysis Module for RoadSurface.

Implements candidate damage extraction via thresholding, morphological refinement,
contour boundary discovery, and geometric region filtering.
"""

from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np


def extract_candidate_mask(
    preprocessed_img: np.ndarray,
    threshold_mode: str = "auto",
    custom_threshold: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts binary candidate damage mask from preprocessed grayscale image.
    Road damage (potholes, structural cracks, asphalt voids) typically exhibits
    lower pixel luminance due to cavity depth and shadowing, surrounded by sharp
    gradient transitions.

    Parameters:
        preprocessed_img (np.ndarray): Denoised and enhanced grayscale image.
        threshold_mode (str): 'auto', 'otsu', 'adaptive', or 'manual'.
        custom_threshold (int, optional): Explicit pixel threshold (0-255).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (raw_threshold_mask, combined_candidate_mask)
    """
    mode = threshold_mode.lower()

    if mode == "manual" or custom_threshold is not None:
        thresh_val = int(custom_threshold if custom_threshold is not None else 80)
        thresh_val = max(0, min(255, thresh_val))
        _, binary = cv2.threshold(
            preprocessed_img, thresh_val, 255, cv2.THRESH_BINARY_INV
        )
    elif mode == "otsu":
        # Otsu's bimodal thresholding
        _, binary = cv2.threshold(
            preprocessed_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    elif mode == "adaptive":
        # Adaptive Gaussian thresholding for non-uniform illumination
        binary = cv2.adaptiveThreshold(
            preprocessed_img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=25,
            C=8,
        )
    else:
        # Default 'auto' mode:
        # Combines statistical luminance depression thresholding with adaptive gradient thresholding.
        # Asphalt pavement has a characteristic modal luminance; potholes and fissure cracks
        # appear as significantly darker depressions (cavity shadowing) and high-contrast boundaries.
        med_lum = float(np.median(preprocessed_img))
        std_lum = float(np.std(preprocessed_img))
        stat_thresh = max(25, int(med_lum - 2.0 * std_lum))
        _, stat_mask = cv2.threshold(
            preprocessed_img, stat_thresh, 255, cv2.THRESH_BINARY_INV
        )
        adaptive_mask = cv2.adaptiveThreshold(
            preprocessed_img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=31,
            C=16,
        )
        binary = cv2.bitwise_or(stat_mask, adaptive_mask)

    return binary, binary


def apply_morphological_refinement(
    binary_mask: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5),
    iterations: int = 1
) -> np.ndarray:
    """
    Applies mathematical morphology (Opening followed by Closing) to:
    1. Eliminate isolated single-pixel gravel noise (MORPH_OPEN).
    2. Bridge adjacent crack segments and seal small cavity gaps (MORPH_CLOSE).

    Parameters:
        binary_mask (np.ndarray): Binary thresholded image (255 for candidates).
        kernel_size (Tuple[int, int]): Structuring element dimensions.
        iterations (int): Repetitions of morphological operations.

    Returns:
        np.ndarray: Cleaned binary damage mask.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    # Opening: Erosion followed by dilation (removes small speckles)
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=iterations)

    # Closing: Dilation followed by erosion (connects fragmented crack contours)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    return closed


def detect_candidate_contours(binary_mask: np.ndarray) -> List[np.ndarray]:
    """
    Extracts external contour polygons from the refined binary damage mask.

    Parameters:
        binary_mask (np.ndarray): 8-bit single-channel binary image.

    Returns:
        List[np.ndarray]: List of detected external contours.
    """
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return list(contours)


def filter_damage_regions(
    contours: List[np.ndarray],
    image_shape: Tuple[int, int],
    min_area: int = 150,
    max_area_ratio: float = 0.35,
    top_margin_ratio: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Filters detected contours based on geometric heuristics to eliminate:
    - Micro-scale road aggregate and surface texture noise (< min_area)
    - Macro-scale shadows from trees/vehicles/overpasses (> max_area_ratio)
    - Sky/horizon artifacts located at the uppermost image border (< top_margin_ratio)

    Parameters:
        contours (List[np.ndarray]): Raw contours from findContours.
        image_shape (Tuple[int, int]): (height, width) of the analyzed image.
        min_area (int): Minimum contour pixel area to be considered valid damage.
        max_area_ratio (float): Maximum fraction of total image area (default 0.35).
        top_margin_ratio (float): Fraction of image top to ignore (default 0.05).

    Returns:
        List[Dict[str, Any]]: List of validated candidate damage region records.
    """
    height, width = image_shape[:2]
    total_image_area = height * width
    max_area = total_image_area * max_area_ratio
    top_cutoff = int(height * top_margin_ratio)

    accepted_regions = []
    region_id = 1

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # 1. Area threshold filtering
        if area < min_area or area > max_area:
            continue

        # 2. Bounding rectangle and aspect ratio
        x, y, w, h = cv2.boundingRect(cnt)

        # 3. Top boundary filtering (rejecting sky / tree horizon if road image is perspective)
        if (y + h) < top_cutoff:
            continue

        # 4. Aspect ratio check (reject degenerate 1-pixel border slivers)
        if w < 3 or h < 3:
            continue
        aspect_ratio = float(w) / float(h)
        if aspect_ratio < 0.05 or aspect_ratio > 20.0:
            continue

        # 5. Centroid calculation using spatial moments
        moments = cv2.moments(cnt)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx = int(x + w / 2)
            cy = int(y + h / 2)

        # 6. Solidity / Extent metrics
        rect_area = w * h
        extent = float(area) / float(rect_area) if rect_area > 0 else 0.0

        accepted_regions.append({
            "id": region_id,
            "contour": cnt,
            "area_pixels": int(area),
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "aspect_ratio": round(aspect_ratio, 2),
            "extent": round(extent, 2),
            "centroid": {"x": cx, "y": cy},
        })
        region_id += 1

    # Sort accepted regions by area descending for consistent reporting
    accepted_regions.sort(key=lambda r: r["area_pixels"], reverse=True)

    # Re-assign sequential IDs after sorting
    for idx, r in enumerate(accepted_regions, start=1):
        r["id"] = idx

    return accepted_regions
