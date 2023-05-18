import unittest
import numpy as np
from tree_health_detection.store_data_structures import *  # import your functions
from pathlib import Path

class TestYourModule(unittest.TestCase):

    def setUp(self):
        # you need to provide test datasets, you might want to use small ones
        self.dataset_path = "path_to_your_test_data"
        self.dataset_path_lidar = "path_to_your_test_lidar_data"
        self.polygon = ... # you need to create a test polygon
        self.output_path = "path_to_output_data"
        self.mask_path = "path_to_mask_data"
        self.status_label = "test_label"
        self.root_dir = Path("path_to_root_dir")
        self.rgb_path = "path_to_rgb_data"
        self.hsi_path = "path_to_hsi_data"
        self.lidar_path = "path_to_lidar_data"
        self.polygon_id = "test_id"
        self.itcs = ... # you need to create a test itcs DataFrame

    def test_extract_data_cube(self):
        binary_mask = extract_data_cube(self.dataset_path, self.polygon, self.output_path, self.mask_path)
        self.assertIsInstance(binary_mask, np.ndarray)
        # add additional checks...

    def test_extract_data_cube_lidar(self):
        extract_data_cube_lidar(self.dataset_path_lidar, self.polygon, self.output_path)
        # check if the file was created
        self.assertTrue(Path(self.output_path).is_file())
        # add additional checks...

    def test_cumulative_linear_stretch(self):
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        test_mask = np.random.choice([True, False], (100, 100), p=[0.2, 0.8])
        stretched_image = cumulative_linear_stretch(test_image, test_mask)
        self.assertIsInstance(stretched_image, np.ndarray)
        # add additional checks...

    def test_png_with_class(self):
        png_with_class(self.dataset_path, self.polygon, None, self.output_path, self.status_label)
        # check if the file was created
        self.assertTrue(Path(self.output_path).is_file())
        # add additional checks...

    def test_process_polygon(self):
        process_polygon(self.polygon, self.root_dir, self.rgb_path, self.hsi_path, self.lidar_path, self.polygon_id, self.itcs)
        # add additional checks...

if __name__ == "__main__":
    unittest.main()
