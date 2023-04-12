import unittest
import numpy as np

from tree_health_detection.src.train_val_test import train, validate, test
from tree_health_detection.src.utils import create_subset, save_dataset, load_dataset, normalize_rgb, clean_hsi_to_0_255_range

class TestTreeHealthDetection(unittest.TestCase):

    def test_clean_hsi_to_0_255_range(self):
        hsi = np.array([
            [10000, 20000, -500],
            [5000, 0, 10000]
        ])
        cleaned_hsi = clean_hsi_to_0_255_range(hsi)
        expected_cleaned_hsi = np.array([
            [255, 0, 0],
            [127.5, 0, 255]
        ])
        np.testing.assert_almost_equal(cleaned_hsi, expected_cleaned_hsi)

    def test_normalize_rgb(self):
        rgb_array = np.array([
            [255, 127, 0],
            [0, 127, 255]
        ], dtype=np.uint8)
        normalized_rgb_array = normalize_rgb(rgb_array)
        expected_normalized_rgb_array = np.array([
            [2.6400, 0.2451, -2.1179],
            [-2.1179, 0.2451, 2.6400]
        ])
        np.testing.assert_almost_equal(normalized_rgb_array, expected_normalized_rgb_array, decimal=4)

    def test_save_and_load_dataset(self):
        dataset = [np.random.rand(5, 5) for _ in range(10)]
        save_dataset(dataset, 'test_dataset.pkl')
        loaded_dataset = load_dataset('test_dataset.pkl')
        for original, loaded in zip(dataset, loaded_dataset):
            np.testing.assert_almost_equal(original, loaded)
        os.remove('test_dataset.pkl')

    def test_create_subset(self):
        data = [1, 2, 3, 4, 5]
        indices = [1, 3]
        subset = create_subset(data, indices)
        self.assertEqual(subset, [2, 4])

if __name__ == '__main__':
    unittest.main()
