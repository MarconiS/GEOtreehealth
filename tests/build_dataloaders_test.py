import pytest
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
from tree_health_detection import MultiModalDataset

class TestMultiModalDataset:
    @pytest.fixture
    def setup(self):
        # Set up the required objects and data
        data = pd.DataFrame({
            'hsi_path': ['test1', 'test2'],
            'rgb_path': ['test1', 'test2'],
            'lidar_path': ['test1', 'test2']
        })
        dataset = MultiModalDataset(data, 'response', 1000)
        return dataset

    def test_len(self, setup):
        assert len(setup) == 2

    def test_getitem(self, setup, mocker):
        # Mock external dependencies
        mocker.patch('numpy.load', return_value=np.zeros((426, 40, 40)))
        mocker.patch('os.path.join', return_value='test')
        sample = setup.__getitem__(0)
        assert sample['hsi'].shape == (426, 40, 40)
        assert sample['rgb'].shape == (3, 400, 400)
        assert sample['lidar'].shape == (1000, 3)

    def test_pad_image(self, setup):
        image = np.zeros((10, 10, 10))
        target_shape = (20, 20, 20)
        padded_image = setup.pad_image(image, target_shape)
        assert padded_image.shape == target_shape

    def test_preprocess(self, setup):
        img_masked = np.random.randint(-10000, 10000, size=(426, 40, 40))
        preprocessed_img = setup.preprocess(img_masked)
        assert preprocessed_img.min() >= 0
        assert preprocessed_img.max() <= 1
        assert preprocessed_img.shape[0] == 274  # After removing bad bands

    def test_normalize_point_cloud(self, setup):
        points = np.random.rand(1000, 3)
        normalized_points = setup.normalize_point_cloud(points.copy())
        centroid = np.mean(normalized_points, axis=0)
        assert np.allclose(centroid, np.zeros(3), atol=1e-7)  # Centroid should be approximately at the origin

    def test_preprocess_rgb(self, setup):
        img_rgb = np.random.randint(-255, 255, size=(3, 400, 400)).astype('float')
        preprocessed_img_rgb = setup.preprocess_rgb(img_rgb)
        assert isinstance(preprocessed_img_rgb, torch.Tensor)
        assert preprocessed_img_rgb.size() == (3, 224, 224)  # After resizing
