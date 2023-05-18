import torch
from torch.autograd import Variable
import unittest
from tree_health_detection import MultiModalNet  # import your model class
from tree_health_detection import SpectralAttentionNetwork  # import your model class
from tree_health_detection import HybridViTWithAttention  # import your model class
from tree_health_detection import DGCNN  # import your model class
import torch.nn as nn
import torch
from torch.autograd import Variable
import unittest

class TestMultiModalNet(unittest.TestCase):

    def setUp(self):
        self.batch_size = 10
        self.num_classes = 2
        self.hsi_out_features = 10
        self.rgb_out_features = 10
        self.lidar_out_features = 10
        self.num_bands = 3
        self.in_channels = 1

        self.model = MultiModalNet(self.in_channels, self.num_classes, self.num_bands,
                                   self.hsi_out_features, self.rgb_out_features, self.lidar_out_features)
        
        self.hsi = Variable(torch.randn(self.batch_size, self.num_bands, 64, 64))
        self.rgb = Variable(torch.randn(self.batch_size, 3, 64, 64))
        self.lidar = Variable(torch.randn(self.batch_size, self.in_channels, 64, 64))

    def test_model_initialization(self):
        self.assertIsNotNone(self.model.hsi_branch)
        self.assertIsNotNone(self.model.rgb_branch)
        self.assertIsNotNone(self.model.lidar_branch)
        self.assertIsNotNone(self.model.fc)

    def test_model_forward_pass(self):
        output = self.model(self.hsi, self.rgb, self.lidar)

        # Check that the output is of the correct type
        self.assertIsInstance(output, torch.Tensor)

        # Check that the output has the correct shape
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))


if __name__ == '__main__':
    unittest.main()
