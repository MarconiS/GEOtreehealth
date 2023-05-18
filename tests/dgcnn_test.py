import torch
from torch.autograd import Variable
import unittest
from tree_health_detection import EdgeConv, DGCNN  # import your model classes

class TestEdgeConv(unittest.TestCase):

    def setUp(self):
        self.batch_size = 10
        self.num_points = 30
        self.k = 20
        self.in_channels = 3
        self.out_channels = 64

        self.edgeconv = EdgeConv(self.in_channels, self.out_channels, self.k)
        self.input_tensor = Variable(torch.randn(self.batch_size, self.in_channels, self.num_points))  

    def test_edgeconv_initialization(self):
        self.assertIsNotNone(self.edgeconv.conv)

    def test_edgeconv_forward_pass(self):
        output = self.edgeconv(self.input_tensor)

        # Check that the output is of the correct type
        self.assertIsInstance(output, torch.Tensor)

        # Check that the output has the correct shape
        self.assertEqual(output.shape, (self.batch_size, self.out_channels, self.num_points))

class TestDGCNN(unittest.TestCase):

    def setUp(self):
        self.batch_size = 10
        self.num_points = 30
        self.k = 20
        self.in_channels = 3
        self.lidar_out_features = 128

        self.dgcnn = DGCNN(self.in_channels, self.lidar_out_features, self.k)
        self.input_tensor = Variable(torch.randn(self.batch_size, self.in_channels, self.num_points))

    def test_dgcnn_initialization(self):
        self.assertIsNotNone(self.dgcnn.transform_net)
        self.assertIsNotNone(self.dgcnn.edgeconv1)
        self.assertIsNotNone(self.dgcnn.edgeconv2)
        self.assertIsNotNone(self.dgcnn.fc1)
        self.assertIsNotNone(self.dgcnn.fc2)

    def test_dgcnn_forward_pass(self):
        output = self.dgcnn(self.input_tensor)

        # Check that the output is of the correct type
        self.assertIsInstance(output, torch.Tensor)

        # Check that the output has the correct shape
        self.assertEqual(output.shape, (self.batch_size, self.lidar_out_features))

if __name__ == '__main__':
    unittest.main()
