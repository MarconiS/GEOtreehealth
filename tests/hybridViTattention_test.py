import torch
from torch.autograd import Variable
import unittest
from tree_health_detection import HybridViTWithAttention, SpatialAttention  # import your model classes

class TestHybridViTWithAttention(unittest.TestCase):

    def setUp(self):
        self.batch_size = 10
        self.num_classes = 3

        self.spatial_attention = SpatialAttention()
        self.hybrid_vit_with_attention = HybridViTWithAttention(self.num_classes)

        self.input_tensor = Variable(torch.randn(self.batch_size, 3, 224, 224))  # assuming input size that fits resnet50

    def test_spatial_attention_initialization(self):
        self.assertIsNotNone(self.spatial_attention.conv1)
        self.assertIsNotNone(self.spatial_attention.sigmoid)

    def test_spatial_attention_forward_pass(self):
        output = self.spatial_attention(self.input_tensor)

        # Check that the output is of the correct type
        self.assertIsInstance(output, torch.Tensor)

        # Check that the output has the correct shape
        self.assertEqual(output.shape, (self.batch_size, 1, 224, 224))

    def test_hybrid_vit_with_attention_initialization(self):
        self.assertIsNotNone(self.hybrid_vit_with_attention.feature_extractor)
        self.assertIsNotNone(self.hybrid_vit_with_attention.spatial_attention)
        self.assertIsNotNone(self.hybrid_vit_with_attention.global_avg_pool)
        self.assertIsNotNone(self.hybrid_vit_with_attention.fc)

    def test_hybrid_vit_with_attention_forward_pass(self):
        output = self.hybrid_vit_with_attention(self.input_tensor)

        # Check that the output is of the correct type
        self.assertIsInstance(output, torch.Tensor)

        # Check that the output has the correct shape
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))


if __name__ == '__main__':
    unittest.main()
