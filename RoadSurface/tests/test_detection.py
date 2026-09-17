"""
Unit tests for damage candidate extraction, morphology, and contour filtering.
"""

import unittest
import numpy as np
import cv2

from src.damage_detection import (
    extract_candidate_mask,
    apply_morphological_refinement,
    detect_candidate_contours,
    filter_damage_regions,
)


class TestDamageDetection(unittest.TestCase):

    def setUp(self):
        # Create a synthetic road patch (500x500)
        self.height, self.width = 500, 500
        self.gray = np.full((self.height, self.width), 130, dtype=np.uint8)

        # Draw a simulated dark pothole at center
        cv2.circle(self.gray, (250, 250), 40, 40, -1)

        # Draw simulated fine gravel specks (tiny noise of 2x2 pixels)
        self.gray[100:102, 100:102] = 30
        self.gray[105:107, 105:107] = 30

    def test_extract_candidate_mask_modes(self):
        for mode in ["auto", "otsu", "adaptive"]:
            _, mask = extract_candidate_mask(self.gray, threshold_mode=mode)
            self.assertEqual(mask.shape, (self.height, self.width))
            self.assertEqual(mask.dtype, np.uint8)
            # All modes must successfully identify damage candidate pixels
            self.assertGreater(int(np.sum(mask > 0)), 0)
        # Verify that auto and otsu capture the deep core cavity at (250, 250)
        _, auto_mask = extract_candidate_mask(self.gray, threshold_mode="auto")
        self.assertEqual(auto_mask[250, 250], 255)

    def test_extract_candidate_mask_manual(self):
        _, mask = extract_candidate_mask(self.gray, threshold_mode="manual", custom_threshold=80)
        self.assertEqual(mask.shape, (self.height, self.width))
        # Pothole (value 40) is < 80, so in THRESH_BINARY_INV it becomes 255
        self.assertEqual(mask[250, 250], 255)
        # Background (value 130) is > 80, so it becomes 0
        self.assertEqual(mask[10, 10], 0)

    def test_apply_morphological_refinement(self):
        binary = np.zeros((200, 200), dtype=np.uint8)
        # Isolated single pixel noise
        binary[50, 50] = 255
        # Solid shape
        cv2.circle(binary, (120, 120), 20, 255, -1)

        refined = apply_morphological_refinement(binary, kernel_size=(5, 5))
        # Isolated single pixel should be eradicated by opening
        self.assertEqual(refined[50, 50], 0)
        # Main solid shape should remain intact
        self.assertEqual(refined[120, 120], 255)

    def test_detect_candidate_contours(self):
        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.rectangle(mask, (50, 50), (120, 120), 255, -1)
        cv2.circle(mask, (220, 220), 25, 255, -1)

        contours = detect_candidate_contours(mask)
        self.assertEqual(len(contours), 2)

    def test_filter_damage_regions(self):
        mask = np.zeros((500, 500), dtype=np.uint8)
        # 1. Tiny speck (area ~9px, should be rejected by min_area=100)
        cv2.rectangle(mask, (50, 50), (53, 53), 255, -1)

        # 2. Legitimate pothole candidate (area ~2000px)
        cv2.circle(mask, (250, 250), 25, 255, -1)

        # 3. Top border artifact (located at y=5, should be rejected by top_margin_ratio)
        cv2.circle(mask, (200, 10), 15, 255, -1)

        contours = detect_candidate_contours(mask)
        regions = filter_damage_regions(
            contours,
            image_shape=(500, 500),
            min_area=100,
            top_margin_ratio=0.08,
        )

        # Only the legitimate central pothole should survive
        self.assertEqual(len(regions), 1)
        region = regions[0]
        self.assertEqual(region["id"], 1)
        self.assertGreater(region["area_pixels"], 1500)
        self.assertIn("centroid", region)
        self.assertAlmostEqual(region["centroid"]["x"], 250, delta=5)
        self.assertAlmostEqual(region["centroid"]["y"], 250, delta=5)


if __name__ == "__main__":
    unittest.main()
