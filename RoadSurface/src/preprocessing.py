"""
Image Preprocessing Module for RoadSurface.

Contains functions for aspect-ratio preserving resizing, color space conversion,
Gaussian smoothing, adaptive contrast enhancement (CLAHE), and edge detection.
"""

from typing import Tuple, Optional
import cv2
import numpy as np


def resize_maintaining_aspect_ratio(
    image: np.ndarray,
    max_dimension: int = 1280
) -> Tuple[np.ndarray, float]:
    """
    Resizes an image so its longest edge does not exceed `max_dimension`,
    maintaining the original aspect ratio without distortion.

    Parameters:
        image (np.ndarray): Original BGR or grayscale image.
        max_dimension (int): Maximum allowable pixel width or height.

    Returns:
        Tuple[np.ndarray, float]: (resized_image, scale_factor)
            where scale_factor = new_size / original_size.
    """
    height, width = image.shape[:2]

    if max(height, width) <= max_dimension:
        return image.copy(), 1.0

    if width >= height:
        scale = max_dimension / float(width)
        new_width = max_dimension
        new_height = int(round(height * scale))
    else:
        scale = max_dimension / float(height)
        new_height = max_dimension
        new_width = int(round(width * scale))

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converts an input image to 8-bit single-channel grayscale if not already grayscale.

    Parameters:
        image (np.ndarray): Input BGR, BGRA, or grayscale image.

    Returns:
        np.ndarray: Single-channel 8-bit grayscale image.
    """
    if len(image.shape) == 2:
        return image.copy()
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_noise_reduction(
    image: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5),
    sigma_x: float = 1.2
) -> np.ndarray:
    """
    Applies Gaussian filtering to suppress high-frequency road texture noise
    and sensor artifacts while preserving prominent distress boundaries.

    Parameters:
        image (np.ndarray): Grayscale input image.
        kernel_size (Tuple[int, int]): Gaussian kernel dimensions (must be odd).
        sigma_x (float): Gaussian standard deviation along the X-axis.

    Returns:
        np.ndarray: Denoised grayscale image.
    """
    # Ensure kernel dimensions are positive odd integers
    kw = kernel_size[0] if kernel_size[0] % 2 != 0 else kernel_size[0] + 1
    kh = kernel_size[1] if kernel_size[1] % 2 != 0 else kernel_size[1] + 1
    return cv2.GaussianBlur(image, (kw, kh), sigmaX=sigma_x)


def enhance_contrast(
    gray_image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    CLAHE enhances local contrast across varying road illumination conditions
    without amplifying uniform background noise.

    Parameters:
        gray_image (np.ndarray): Grayscale image.
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (Tuple[int, int]): Size of grid for histogram equalization.

    Returns:
        np.ndarray: Contrast-enhanced grayscale image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_image)


def detect_edges(
    gray_image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150
) -> np.ndarray:
    """
    Applies Canny edge detection with configurable hysteresis thresholds.

    Parameters:
        gray_image (np.ndarray): Grayscale, smoothed image.
        low_threshold (int): Lower hysteresis threshold.
        high_threshold (int): Upper hysteresis threshold.

    Returns:
        np.ndarray: Binary edge map.
    """
    return cv2.Canny(gray_image, threshold1=low_threshold, threshold2=high_threshold)


def run_preprocessing_pipeline(
    image: np.ndarray,
    max_dimension: int = 1280,
    apply_clahe: bool = True
) -> dict:
    """
    Executes the full preprocessing sequence and returns processed images along
    with intermediate states for visualization.

    Parameters:
        image (np.ndarray): Raw input BGR image.
        max_dimension (int): Max dimension for resizing.
        apply_clahe (bool): Whether to apply CLAHE contrast enhancement.

    Returns:
        dict: Containing 'resized', 'scale', 'gray', 'blurred', 'enhanced', 'edges'.
    """
    resized, scale = resize_maintaining_aspect_ratio(image, max_dimension=max_dimension)
    gray = to_grayscale(resized)
    blurred = apply_noise_reduction(gray, kernel_size=(5, 5), sigma_x=1.2)

    if apply_clahe:
        enhanced = enhance_contrast(blurred, clip_limit=2.0)
    else:
        enhanced = blurred.copy()

    edges = detect_edges(enhanced, low_threshold=50, high_threshold=150)

    return {
        "resized": resized,
        "scale": scale,
        "gray": gray,
        "blurred": blurred,
        "enhanced": enhanced,
        "edges": edges,
    }
