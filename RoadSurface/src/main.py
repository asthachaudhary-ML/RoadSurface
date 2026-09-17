"""
RoadSurface CLI Application Entry Point.

Usage:
    python src/main.py --input data/input/sample_road.jpg
    python src/main.py --input path/to/road.jpg --output data/output --threshold auto --min-area 150 --save-intermediate
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

# Support direct invocation from repository root or subfolder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    validate_image_path,
    save_json_report,
    print_cli_banner,
    print_stage_progress,
    print_results_summary,
)
from src.preprocessing import run_preprocessing_pipeline
from src.damage_detection import (
    extract_candidate_mask,
    apply_morphological_refinement,
    detect_candidate_contours,
    filter_damage_regions,
)
from src.severity import analyze_severity
from src.visualization import draw_annotated_image, save_intermediate_stages


def parse_arguments() -> argparse.Namespace:
    """Configures and parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RoadSurface: Road Damage Detection and Severity Analysis using Computer Vision",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/input/sample_road.jpg",
        help="Path to input road image (JPG, JPEG, PNG)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/output",
        help="Directory to save annotated image and JSON report",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=str,
        default="auto",
        help="Candidate extraction threshold mode: 'auto', 'otsu', 'adaptive', or integer value (e.g. '85')",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=150,
        help="Minimum pixel area threshold to filter small gravel and texture noise",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate pipeline stage images (grayscale, blurred, edges, threshold, morphology)",
    )
    parser.add_argument(
        "--clahe",
        action="store_true",
        help="Enable CLAHE contrast enhancement (default: disabled to avoid amplifying uniform pavement noise)",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=1280,
        help="Maximum width or height for resizing while preserving aspect ratio",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    # Step 1: Input Validation
    try:
        image_path = validate_image_path(args.input)
    except (FileNotFoundError, ValueError) as err:
        print(str(err), file=sys.stderr)
        return 1

    # Load image in BGR format
    raw_bgr = cv2.imread(str(image_path))
    if raw_bgr is None:
        print("Error: Unable to read image.", file=sys.stderr)
        return 1

    orig_height, orig_width = raw_bgr.shape[:2]

    # Display Header Banner
    print_cli_banner()

    # Step 2: Image Preprocessing Pipeline
    prep_data = run_preprocessing_pipeline(
        raw_bgr,
        max_dimension=args.max_dimension,
        apply_clahe=args.clahe,
    )
    resized_bgr = prep_data["resized"]
    enhanced_gray = prep_data["enhanced"]

    # Step 3: Candidate Damage Extraction
    threshold_mode = args.threshold
    custom_thresh_val = None
    if threshold_mode.isdigit():
        custom_thresh_val = int(threshold_mode)
        threshold_mode = "manual"

    _, candidate_mask = extract_candidate_mask(
        enhanced_gray,
        threshold_mode=threshold_mode,
        custom_threshold=custom_thresh_val,
    )

    # Step 4: Morphological Refinement
    refined_mask = apply_morphological_refinement(
        candidate_mask,
        kernel_size=(5, 5),
        iterations=1,
    )

    # Step 5: Contour Detection & Filtering
    raw_contours = detect_candidate_contours(refined_mask)
    accepted_regions = filter_damage_regions(
        raw_contours,
        image_shape=resized_bgr.shape,
        min_area=args.min_area,
    )

    # Display progress
    print_stage_progress(
        image_name=image_path.name,
        resolution=(orig_width, orig_height),
        preprocessed=True,
        extracted=True,
        analyzed=True,
    )

    # Step 6: Severity & Quantitative Metrics Analysis
    severity_info = analyze_severity(accepted_regions, resized_bgr.shape)

    # Step 7: Output Generation & Visualization
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_image = draw_annotated_image(resized_bgr, accepted_regions, severity_info)
    annotated_path = output_dir / "annotated_road.jpg"
    cv2.imwrite(str(annotated_path), annotated_image)

    # JSON report structure
    serializable_regions = [
        {
            "id": r["id"],
            "area_pixels": r["area_pixels"],
            "x": r["x"],
            "y": r["y"],
            "width": r["width"],
            "height": r["height"],
            "aspect_ratio": r["aspect_ratio"],
            "extent": r["extent"],
            "centroid": r["centroid"],
        }
        for r in accepted_regions
    ]

    report_payload = {
        "input_image": image_path.name,
        "image_width": orig_width,
        "image_height": orig_height,
        "processed_width": resized_bgr.shape[1],
        "processed_height": resized_bgr.shape[0],
        "damage_regions": severity_info["region_count"],
        "damage_percentage": severity_info["damage_percentage"],
        "severity_score": severity_info["severity_score"],
        "severity_level": severity_info["severity_level"],
        "metrics": {
            "road_area_pixels": severity_info["road_area_pixels"],
            "damage_area_pixels": severity_info["damage_area_pixels"],
            "largest_region_pixels": severity_info["largest_region_pixels"],
            "largest_region_percentage": severity_info["largest_region_percentage"],
        },
        "regions": serializable_regions,
    }

    json_path = output_dir / "analysis.json"
    save_json_report(report_payload, json_path)

    # Step 8: Save Intermediate Stages (if requested)
    if args.save_intermediate:
        inter_dir = output_dir / "intermediate"
        # Render a contour debug preview
        contour_preview = resized_bgr.copy()
        cv2.drawContours(contour_preview, raw_contours, -1, (0, 255, 255), 1)

        stages_dict = {
            "grayscale": prep_data["gray"],
            "blurred": prep_data["blurred"],
            "enhanced": prep_data["enhanced"],
            "edges": prep_data["edges"],
            "threshold": candidate_mask,
            "morphology": refined_mask,
            "contours": contour_preview,
        }
        save_intermediate_stages(stages_dict, inter_dir)

    # Step 9: Print CLI Summary
    print_results_summary(
        damage_regions=severity_info["region_count"],
        damage_pct=severity_info["damage_percentage"],
        largest_region_pct=severity_info["largest_region_percentage"],
        severity_score=severity_info["severity_score"],
        severity_level=severity_info["severity_level"],
        annotated_image_path=annotated_path,
        json_report_path=json_path,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
