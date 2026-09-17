"""
Batch Experiment and Evaluation Script for RoadSurface.

Evaluates road images in data/input/, calculates damage metrics,
measures execution latency, and exports verifiable results to results.csv.
"""

import sys
import time
import csv
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.preprocessing import run_preprocessing_pipeline
from src.damage_detection import (
    extract_candidate_mask,
    apply_morphological_refinement,
    detect_candidate_contours,
    filter_damage_regions,
)
from src.severity import analyze_severity
from src.visualization import draw_annotated_image
from src.utils import SUPPORTED_EXTENSIONS


def run_batch_evaluation(input_dir: Path, output_dir: Path, csv_output: Path) -> None:
    """Runs pipeline over all images in input_dir and saves metrics to CSV."""
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not image_files:
        print(f"No valid image files found in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    print("\n" + "=" * 80)
    print("ROADSURFACE BATCH EXPERIMENT & BENCHMARKING")
    print("=" * 80)
    print(f"{'Image Name':<20} | {'Regions':<8} | {'Damage %':<10} | {'Score':<8} | {'Level':<8} | {'Time (ms)':<10}")
    print("-" * 80)

    for img_path in sorted(image_files):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        start_t = time.perf_counter()

        # Execute Pipeline (clahe=False to avoid noise amplification on uniform pavement)
        prep = run_preprocessing_pipeline(img_bgr, max_dimension=1280, apply_clahe=False)
        _, candidate_mask = extract_candidate_mask(prep["enhanced"], threshold_mode="auto")
        refined = apply_morphological_refinement(candidate_mask, kernel_size=(5, 5))
        raw_cnts = detect_candidate_contours(refined)
        regions = filter_damage_regions(raw_cnts, prep["resized"].shape, min_area=150)
        sev = analyze_severity(regions, prep["resized"].shape)

        annotated = draw_annotated_image(prep["resized"], regions, sev)
        out_img_path = output_dir / f"annotated_{img_path.stem}.jpg"
        cv2.imwrite(str(out_img_path), annotated)

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        row = {
            "image_name": img_path.name,
            "width": img_bgr.shape[1],
            "height": img_bgr.shape[0],
            "detected_regions": sev["region_count"],
            "damage_percentage": sev["damage_percentage"],
            "largest_region_percentage": sev["largest_region_percentage"],
            "severity_score": sev["severity_score"],
            "severity_level": sev["severity_level"],
            "processing_time_ms": round(elapsed_ms, 2),
        }
        results.append(row)

        print(
            f"{img_path.name:<20} | "
            f"{row['detected_regions']:<8} | "
            f"{row['damage_percentage']:<10.2f} | "
            f"{row['severity_score']:<8.1f} | "
            f"{row['severity_level']:<8} | "
            f"{row['processing_time_ms']:<10.2f}"
        )

    print("=" * 80)

    # Save to CSV
    csv_fields = [
        "image_name",
        "width",
        "height",
        "detected_regions",
        "damage_percentage",
        "largest_region_percentage",
        "severity_score",
        "severity_level",
        "processing_time_ms",
    ]
    with open(csv_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nBatch results saved successfully to: {csv_output}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
    in_dir = repo_root / "data" / "input"
    out_dir = repo_root / "data" / "output"
    csv_file = repo_root / "results.csv"
    run_batch_evaluation(in_dir, out_dir, csv_file)
