"""
Unit tests for image preprocessing module.
"""

import unittest
import numpy as np
import cv2

from src.preprocessing import (
    resize_maintaining_aspect_ratio,
    to_grayscale,
    apply_noise_reduction,
    enhance_contrast,
    detect_edges,
    run_preprocessing_pipeline,
)


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Create synthetic test BGR image (width=1600, height=800)
        self.large_img = np.full((800, 1600, 3), 120, dtype=np.uint8)
        # Add some patterns
        cv2.circle(self.large_img, (800, 400), 50, (30, 30, 30), -1)

        # Create smaller image (width=600, height=400)
        self.small_img = np.full((400, 600, 3), 120, dtype=np.uint8)

    def test_resize_maintaining_aspect_ratio_large(self):
        resized, scale = resize_maintaining_aspect_ratio(self.large_img, max_dimension=1280)
        h, w = resized.shape[:2]
        self.assertEqual(w, 1280)
        self.assertEqual(h, 640)
        self.assertAlmostEqual(scale, 1280 / 1600, places=4)

    def test_resize_maintaining_aspect_ratio_small(self):
        resized, scale = resize_maintaining_aspect_ratio(self.small_img, max_dimension=1280)
        h, w = resized.shape[:2]
        self.assertEqual(w, 600)
        self.assertEqual(h, 400)
        self.assertEqual(scale, 1.0)

    def test_to_grayscale(self):
        gray = to_grayscale(self.large_img)
        self.assertEqual(len(gray.shape), 2)
        self.assertEqual(gray.shape, (800, 1600))
        # Idempotency test (converting already grayscale image)
        gray_again = to_grayscale(gray)
        self.assertEqual(len(gray_again.shape), 2)
        np.testing.assert_array_equal(gray, gray_again)

    def test_apply_noise_reduction(self):
        gray = to_grayscale(self.small_img)
        # Inject impulse noise (isolated spike)
        noisy = gray.copy()
        noisy[102, 102] = 255
        denoised = apply_noise_reduction(noisy, kernel_size=(5, 5), sigma_x=1.0)
        self.assertEqual(denoised.shape, gray.shape)
        # Peak value should be smoothed down significantly from 255
        self.assertLess(denoised[102, 102], 255)

    def test_enhance_contrast(self):
        gray = to_grayscale(self.small_img)
        enhanced = enhance_contrast(gray, clip_limit=2.0)
        self.assertEqual(enhanced.shape, gray.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_detect_edges(self):
        gray = to_grayscale(self.small_img)
        edges = detect_edges(gray, low_threshold=50, high_threshold=150)
        self.assertEqual(edges.shape, gray.shape)
        # Edges must be binary: values in {0, 255}
        unique_vals = np.unique(edges)
        for val in unique_vals:
            self.assertIn(val, [0, 255])

    def test_run_preprocessing_pipeline(self):
        pipeline_output = run_preprocessing_pipeline(self.large_img, max_dimension=1280)
        expected_keys = {"resized", "scale", "gray", "blurred", "enhanced", "edges"}
        self.assertTrue(expected_keys.issubset(pipeline_output.keys()))
        self.assertEqual(pipeline_output["resized"].shape[1], 1280)


if __name__ == "__main__":
    unittest.main()
