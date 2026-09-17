"""
Severity Calculation and Quantitative Metrics Module for RoadSurface.

Computes visible damage area, damage percentage relative to the visible road surface,
and an interpretable 0-100 severity index mapped to LOW, MEDIUM, or HIGH classifications.
"""

from typing import List, Dict, Any, Tuple


# Severity Level Thresholds (Heuristic Project Standards)
SEVERITY_THRESHOLD_LOW_MAX = 30.0
SEVERITY_THRESHOLD_MED_MAX = 60.0


def calculate_damage_metrics(
    regions: List[Dict[str, Any]],
    image_shape: Tuple[int, int]
) -> Dict[str, Any]:
    """
    Quantifies the total and individual spatial extent of detected damage in image space.

    Parameters:
        regions (List[Dict[str, Any]]): Filtered candidate damage regions.
        image_shape (Tuple[int, int]): (height, width) of the analyzed road frame.

    Returns:
        Dict[str, Any]: Damage area metrics and surface proportions.
    """
    height, width = image_shape[:2]
    road_area_pixels = height * width

    damage_area_pixels = sum(r["area_pixels"] for r in regions)
    region_count = len(regions)

    if road_area_pixels > 0:
        damage_percentage = (damage_area_pixels / float(road_area_pixels)) * 100.0
    else:
        damage_percentage = 0.0

    if regions:
        largest_region_pixels = max(r["area_pixels"] for r in regions)
        largest_region_percentage = (largest_region_pixels / float(road_area_pixels)) * 100.0
    else:
        largest_region_pixels = 0
        largest_region_percentage = 0.0

    return {
        "road_area_pixels": road_area_pixels,
        "damage_area_pixels": damage_area_pixels,
        "damage_percentage": round(damage_percentage, 2),
        "region_count": region_count,
        "largest_region_pixels": largest_region_pixels,
        "largest_region_percentage": round(largest_region_percentage, 2),
    }


def compute_severity_score(
    damage_percentage: float,
    region_count: int,
    largest_region_percentage: float
) -> float:
    """
    Computes an interpretable composite severity score between 0.0 and 100.0.

    Heuristic Formula:
        Score = w1 * S_area + w2 * S_count + w3 * S_worst

        Where:
        - S_area: Scaled damage percentage (saturates at 15.0% surface damage)
        - S_count: Scaled defect count (saturates at 8 distinct damage clusters)
        - S_worst: Scaled largest individual crater/crack percentage (saturates at 6.0%)

        Weights:
        - w1 = 0.50 (overall surface degradation)
        - w2 = 0.25 (defect proliferation/frequency)
        - w3 = 0.25 (structural severity of the worst single defect)

    Parameters:
        damage_percentage (float): Percentage of road surface affected.
        region_count (int): Number of detected candidate regions.
        largest_region_percentage (float): Area percentage of largest single defect.

    Returns:
        float: Composite score constrained to [0.0, 100.0].
    """
    if region_count == 0 or damage_percentage <= 0.0:
        return 0.0

    # Normalization components (bounded to 100.0 each)
    # 15% road surface damage represents severe structural disintegration in an image
    s_area = min(100.0, (damage_percentage / 15.0) * 100.0)

    # 8 or more distinct damage regions represents widespread road cracking
    s_count = min(100.0, (float(region_count) / 8.0) * 100.0)

    # A single pothole or crater occupying >= 6% of the frame is an acute hazard
    s_worst = min(100.0, (largest_region_percentage / 6.0) * 100.0)

    # Weighted combination
    composite_score = (0.50 * s_area) + (0.25 * s_count) + (0.25 * s_worst)

    return round(float(min(100.0, max(0.0, composite_score))), 1)


def classify_severity_level(severity_score: float) -> str:
    """
    Maps numeric severity score to categorical engineering level:
    - 0.0 to 30.0   : LOW (Minor cosmetic defects, small isolated cracks)
    - 30.1 to 60.0  : MEDIUM (Noticeable potholes, moderate cluster deterioration)
    - 60.1 to 100.0 : HIGH (Severe roadway disintegration, large hazardous craters)

    Parameters:
        severity_score (float): Composite score in [0.0, 100.0].

    Returns:
        str: 'LOW', 'MEDIUM', or 'HIGH'.
    """
    if severity_score <= SEVERITY_THRESHOLD_LOW_MAX:
        return "LOW"
    elif severity_score <= SEVERITY_THRESHOLD_MED_MAX:
        return "MEDIUM"
    else:
        return "HIGH"


def analyze_severity(
    regions: List[Dict[str, Any]],
    image_shape: Tuple[int, int]
) -> Dict[str, Any]:
    """
    High-level orchestrator returning both raw metrics and classified severity.
    """
    metrics = calculate_damage_metrics(regions, image_shape)
    score = compute_severity_score(
        metrics["damage_percentage"],
        metrics["region_count"],
        metrics["largest_region_percentage"],
    )
    level = classify_severity_level(score)

    return {
        **metrics,
        "severity_score": score,
        "severity_level": level,
    }
