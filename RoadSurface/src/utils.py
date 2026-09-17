"""
Utility functions for RoadSurface: path validation, formatted terminal display,
and structured JSON report export.
"""

from pathlib import Path
import json
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_image_path(image_path_str: str) -> Path:
    """
    Validates that the input image path exists, has a valid image extension,
    and is readable by OpenCV.

    Parameters:
        image_path_str (str): Path to input image.

    Returns:
        Path: Validated pathlib.Path object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file extension is unsupported or file cannot be decoded.
    """
    path = Path(image_path_str)

    if not path.exists():
        raise FileNotFoundError("Error: Input image not found.")

    if not path.is_file():
        raise ValueError(f"Error: Specified path '{path}' is not a valid file.")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Error: Unsupported image format '{path.suffix}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # Verify that the image can actually be decoded
    test_img = cv2.imread(str(path))
    if test_img is None or test_img.size == 0:
        raise ValueError("Error: Unable to read image.")

    return path


def save_json_report(report_dict: Dict[str, Any], output_path: Path) -> None:
    """
    Saves the structured analysis dictionary as a human-readable JSON file.

    Parameters:
        report_dict (dict): Complete analysis metrics and regions.
        output_path (Path): Target file path for the JSON output.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)


def print_cli_banner() -> None:
    """Prints the application header banner."""
    print("=========================================")
    print("        ROADSURFACE ANALYZER             ")
    print("=========================================")


def print_stage_progress(
    image_name: str,
    resolution: Tuple[int, int],
    preprocessed: bool = True,
    extracted: bool = True,
    analyzed: bool = True,
) -> None:
    """
    Prints standard pipeline stage status.

    Parameters:
        image_name (str): Basename of the input image.
        resolution (tuple): (width, height)
    """
    print(f"Input image        : {image_name}")
    print(f"Image resolution   : {resolution[0]} x {resolution[1]}\n")
    print(f"Preprocessing      : {'Completed' if preprocessed else 'Skipped'}")
    print(f"Damage extraction  : {'Completed' if extracted else 'Skipped'}")
    print(f"Contour analysis   : {'Completed' if analyzed else 'Skipped'}")


def print_results_summary(
    damage_regions: int,
    damage_pct: float,
    largest_region_pct: float,
    severity_score: float,
    severity_level: str,
    annotated_image_path: Path,
    json_report_path: Path,
) -> None:
    """
    Prints formatted final results matching course evaluation requirements.
    """
    print("\n-----------------------------------------")
    print("RESULTS")
    print("-----------------------------------------")
    print(f"Damage regions     : {damage_regions}")
    print(f"Damaged area       : {damage_pct:.2f}%")
    print(f"Largest region     : {largest_region_pct:.2f}%")
    print(f"Severity score     : {severity_score:.1f} / 100")
    print(f"Severity level     : {severity_level}")
    print("-----------------------------------------\n")
    print("Annotated image:")
    print(f"{annotated_image_path}\n")
    print("Analysis report:")
    print(f"{json_report_path}")
    print("=========================================")
