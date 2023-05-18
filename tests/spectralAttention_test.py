import torch
from torch.autograd import Variable
import unittest
from tree_health_detection import SpectralAttentionLayer, SpectralAttentionNetwork  # import your model classes

class TestSpectralAttention(unittest.TestCase):

    def setUp(self):
        self.batch_size = 10
        self.hsi_out_features = 10
        self.num_bands = 3

        self.spectral_attention_layer = SpectralAttentionLayer(self.hsi_out_features)
        self.spectral_attention_network = SpectralAttentionNetwork(self.num_bands, self.hsi_out_features)

        self.input_tensor = Variable(torch.randn(self.batch_size, self.num_bands, 64, 64))

    def test_spectral_attention_layer_initialization(self):
        self.assertIsNotNone(self.spectral_attention_layer.avg_pool)
        self.assertIsNotNone(self.spectral_attention_layer.max_pool)
        self.assertIsNotNone(self.spectral_attention_layer.fc)
        self.assertIsNotNone(self.spectral_attention_layer.sigmoid)

    def test_spectral_attention_layer_forward_pass(self):
        output = self.spectral_attention_layer(self.input_tensor)

        # Check that the output is of the correct type
        self.assertIsInstance(output, torch.Tensor)

        # Check that the output has the correct shape
        self.assertEqual(output.shape, self.input_tensor.shape)

    def test_spectral_attention_network_initialization(self):
        self.assertIsNotNone(self.spectral_attention_network.conv1)
        self.assertIsNotNone(self.spectral_attention_network.bn1)
        self.assertIsNotNone(self.spectral_attention_network.relu)
        self.assertIsNotNone(self.spectral_attention_network.spectral_attention)

    def test_spectral_attention_network_forward_pass(self):
        output = self.spectral_attention_network(self.input_tensor)

        # Check that the output is of the correct type
        self.assertIsInstance(output, torch.Tensor)

        # Check that the output has the correct shape
        self.assertEqual(output.shape, self.input_tensor.shape)


if __name__ == '__main__':
    unittest.main()
