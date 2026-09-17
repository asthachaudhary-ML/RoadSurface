"""
Unit tests for utility functions, error handling, and JSON serialization.
"""

import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
import cv2

from src.utils import validate_image_path, save_json_report


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)

        # Create a valid test image
        self.valid_img_path = self.temp_dir_path / "valid.jpg"
        test_img = np.full((100, 100, 3), 150, dtype=np.uint8)
        cv2.imwrite(str(self.valid_img_path), test_img)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_validate_image_path_success(self):
        resolved = validate_image_path(str(self.valid_img_path))
        self.assertTrue(resolved.exists())
        self.assertEqual(resolved.suffix.lower(), ".jpg")

    def test_validate_image_path_not_found(self):
        non_existent = self.temp_dir_path / "ghost_image.jpg"
        with self.assertRaises(FileNotFoundError) as ctx:
            validate_image_path(str(non_existent))
        self.assertEqual(str(ctx.exception), "Error: Input image not found.")

    def test_validate_image_path_unsupported_extension(self):
        invalid_ext_file = self.temp_dir_path / "notes.txt"
        invalid_ext_file.write_text("not an image")
        with self.assertRaises(ValueError) as ctx:
            validate_image_path(str(invalid_ext_file))
        self.assertIn("Error: Unsupported image format", str(ctx.exception))

    def test_validate_image_path_corrupted(self):
        corrupted_file = self.temp_dir_path / "bad.jpg"
        corrupted_file.write_bytes(b"corrupted binary data that opencv cannot parse")
        with self.assertRaises(ValueError) as ctx:
            validate_image_path(str(corrupted_file))
        self.assertEqual(str(ctx.exception), "Error: Unable to read image.")

    def test_save_json_report(self):
        target_json = self.temp_dir_path / "output" / "test_report.json"
        sample_data = {
            "input_image": "test.jpg",
            "damage_regions": 2,
            "damage_percentage": 4.5,
            "severity_score": 28.0,
            "severity_level": "LOW",
        }
        save_json_report(sample_data, target_json)

        self.assertTrue(target_json.exists())
        with open(target_json, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(loaded["damage_regions"], 2)
        self.assertEqual(loaded["severity_level"], "LOW")


if __name__ == "__main__":
    unittest.main()
