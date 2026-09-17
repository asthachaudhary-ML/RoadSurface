"""
Unit tests for quantitative damage metrics and severity scoring logic.
"""

import unittest
from src.severity import (
    calculate_damage_metrics,
    compute_severity_score,
    classify_severity_level,
    analyze_severity,
)


class TestSeverity(unittest.TestCase):

    def test_damage_metrics_calculation(self):
        # Image resolution: 1000 x 1000 = 1,000,000 pixels
        image_shape = (1000, 1000)
        regions = [
            {"id": 1, "area_pixels": 50000},
            {"id": 2, "area_pixels": 20000},
            {"id": 3, "area_pixels": 10000},
        ]
        metrics = calculate_damage_metrics(regions, image_shape)

        self.assertEqual(metrics["road_area_pixels"], 1000000)
        self.assertEqual(metrics["damage_area_pixels"], 80000)
        self.assertEqual(metrics["region_count"], 3)
        self.assertEqual(metrics["largest_region_pixels"], 50000)
        # 80,000 / 1,000,000 = 8.0%
        self.assertAlmostEqual(metrics["damage_percentage"], 8.0, places=2)
        # 50,000 / 1,000,000 = 5.0%
        self.assertAlmostEqual(metrics["largest_region_percentage"], 5.0, places=2)

    def test_damage_metrics_empty_regions(self):
        metrics = calculate_damage_metrics([], (800, 1200))
        self.assertEqual(metrics["damage_area_pixels"], 0)
        self.assertEqual(metrics["damage_percentage"], 0.0)
        self.assertEqual(metrics["region_count"], 0)
        self.assertEqual(metrics["largest_region_percentage"], 0.0)

    def test_severity_score_bounds_and_zero(self):
        # Clean road with zero damage
        zero_score = compute_severity_score(0.0, 0, 0.0)
        self.assertEqual(zero_score, 0.0)

        # Extreme damage exceeding saturation limits
        extreme_score = compute_severity_score(45.0, 30, 20.0)
        self.assertEqual(extreme_score, 100.0)

        # Moderate realistic damage (e.g. 5% damage, 4 regions, 2.5% largest)
        mod_score = compute_severity_score(5.0, 4, 2.5)
        self.assertTrue(0.0 <= mod_score <= 100.0)

    def test_classify_severity_level(self):
        self.assertEqual(classify_severity_level(0.0), "LOW")
        self.assertEqual(classify_severity_level(25.0), "LOW")
        self.assertEqual(classify_severity_level(30.0), "LOW")

        self.assertEqual(classify_severity_level(30.1), "MEDIUM")
        self.assertEqual(classify_severity_level(48.5), "MEDIUM")
        self.assertEqual(classify_severity_level(60.0), "MEDIUM")

        self.assertEqual(classify_severity_level(60.1), "HIGH")
        self.assertEqual(classify_severity_level(85.0), "HIGH")
        self.assertEqual(classify_severity_level(100.0), "HIGH")

    def test_analyze_severity_orchestrator(self):
        regions = [{"id": 1, "area_pixels": 40000}]
        result = analyze_severity(regions, (1000, 1000))

        self.assertIn("damage_percentage", result)
        self.assertIn("severity_score", result)
        self.assertIn("severity_level", result)
        self.assertIn(result["severity_level"], ["LOW", "MEDIUM", "HIGH"])


if __name__ == "__main__":
    unittest.main()
