"""
Visualization and Image Annotation Module for RoadSurface.

Renders high-contrast bounding boxes, contour boundaries, region labels,
and an informational HUD summary overlay on the road image.
Also exports intermediate pipeline stages for educational and experimental demonstration.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


# Color palettes (BGR format for OpenCV)
COLOR_SEVERITY_MAP = {
    "LOW": (50, 205, 50),     # Lime Green
    "MEDIUM": (0, 165, 255),   # Vibrant Orange
    "HIGH": (0, 0, 230),       # High-intensity Red
}
COLOR_BOX = (0, 220, 255)       # Yellow-cyan for bounding box
COLOR_TEXT = (255, 255, 255)    # White text
COLOR_BG_HUD = (30, 30, 30)     # Dark slate background for HUD


def draw_annotated_image(
    base_image: np.ndarray,
    regions: List[Dict[str, Any]],
    severity_info: Dict[str, Any]
) -> np.ndarray:
    """
    Renders visual damage markings and a non-intrusive HUD summary banner.

    Parameters:
        base_image (np.ndarray): Original or resized BGR image.
        regions (List[Dict[str, Any]]): List of validated damage regions.
        severity_info (Dict[str, Any]): Metrics and severity classification.

    Returns:
        np.ndarray: Annotated BGR image.
    """
    annotated = base_image.copy()
    overlay = base_image.copy()

    severity_level = severity_info.get("severity_level", "LOW")
    accent_color = COLOR_SEVERITY_MAP.get(severity_level, (0, 255, 0))

    # 1. Draw contour fills (semi-transparent) and contour outlines
    for r in regions:
        cnt = r["contour"]
        # Semi-transparent polygon fill
        cv2.drawContours(overlay, [cnt], -1, accent_color, thickness=-1)
        # Crisp boundary stroke
        cv2.drawContours(annotated, [cnt], -1, accent_color, thickness=2)

    # Blend semi-transparent damage fills
    alpha = 0.25
    cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0, annotated)

    # 2. Draw bounding boxes and region tags
    for r in regions:
        x, y, w, h = r["x"], r["y"], r["width"], r["height"]
        reg_id = r["id"]
        area_px = r["area_pixels"]

        # Bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), COLOR_BOX, 1, lineType=cv2.LINE_AA)

        # Region label badge
        label = f"Damage #{reg_id} ({area_px}px)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness = 1

        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        tag_y = max(y - 5, label_h + 5)

        # Tag background
        cv2.rectangle(
            annotated,
            (x, tag_y - label_h - 4),
            (x + label_w + 6, tag_y + baseline - 2),
            (20, 20, 20),
            -1,
        )
        # Tag border
        cv2.rectangle(
            annotated,
            (x, tag_y - label_h - 4),
            (x + label_w + 6, tag_y + baseline - 2),
            COLOR_BOX,
            1,
        )
        # Tag text
        cv2.putText(
            annotated,
            label,
            (x + 3, tag_y - 2),
            font,
            font_scale,
            COLOR_TEXT,
            thickness,
            cv2.LINE_AA,
        )

    # 3. Draw Top-Right Heads-Up Display (HUD) Banner
    annotated = draw_hud_banner(annotated, severity_info, accent_color)

    return annotated


def draw_hud_banner(
    image: np.ndarray,
    severity_info: Dict[str, Any],
    accent_color: Tuple[int, int, int]
) -> np.ndarray:
    """
    Renders a compact, translucent information card in the upper corner
    displaying regions count, damage area %, severity score and level.
    """
    img_h, img_w = image.shape[:2]

    card_w = 320
    card_h = 130
    margin = 15

    x1 = img_w - card_w - margin
    y1 = margin
    x2 = img_w - margin
    y2 = margin + card_h

    # Ensure bounds stay within image
    if x1 < 0 or y1 < 0:
        return image

    hud_overlay = image.copy()
    cv2.rectangle(hud_overlay, (x1, y1), (x2, y2), COLOR_BG_HUD, -1)
    cv2.addWeighted(hud_overlay, 0.78, image, 0.22, 0, image)

    # Card border with severity accent color
    cv2.rectangle(image, (x1, y1), (x2, y2), accent_color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title header
    cv2.putText(image, "RoadSurface Analysis", (x1 + 14, y1 + 25), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(image, (x1 + 14, y1 + 33), (x2 - 14, y1 + 33), (80, 80, 80), 1)

    # Metrics lines
    reg_count = severity_info.get("region_count", 0)
    dmg_pct = severity_info.get("damage_percentage", 0.0)
    sev_score = severity_info.get("severity_score", 0.0)
    sev_level = severity_info.get("severity_level", "LOW")

    cv2.putText(image, f"Regions     : {reg_count}", (x1 + 14, y1 + 56), font, 0.44, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(image, f"Damage Area : {dmg_pct:.2f}%", (x1 + 14, y1 + 77), font, 0.44, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(image, f"Score       : {sev_score:.1f} / 100", (x1 + 14, y1 + 98), font, 0.44, (220, 220, 220), 1, cv2.LINE_AA)

    # Highlighted Severity Level
    cv2.putText(image, f"Severity    : {sev_level}", (x1 + 14, y1 + 119), font, 0.48, accent_color, 2, cv2.LINE_AA)

    return image


def save_intermediate_stages(
    stages: Dict[str, np.ndarray],
    output_directory: Path
) -> List[Path]:
    """
    Saves intermediate pipeline transformation images into the specified directory.

    Supported stages:
        - grayscale.jpg
        - blurred.jpg
        - enhanced.jpg
        - edges.jpg
        - threshold.jpg
        - morphology.jpg
        - raw_contours.jpg

    Parameters:
        stages (Dict[str, np.ndarray]): Mapping of stage names to image arrays.
        output_directory (Path): Target directory for intermediate files.

    Returns:
        List[Path]: Paths of saved intermediate files.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for name, img in stages.items():
        if img is None:
            continue
        file_path = output_directory / f"{name}.jpg"
        cv2.imwrite(str(file_path), img)
        saved_paths.append(file_path)

    return saved_paths
