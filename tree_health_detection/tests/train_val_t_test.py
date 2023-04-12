import unittest
import torch
from torch.utils.data import Dataset
from tree_health_detection.src.train_val_t import train, validate, test


class DummyDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rgb = torch.randn(3, 224, 224)
        hs = torch.randn(3, 224, 224)
        lidar = torch.randn(1, 224, 224)
        label = torch.randint(0, 10, (1,)).item()
        return rgb, hs, lidar, label


class TestFunctions(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Sequential(torch.nn.Linear(3 * 224 * 224, 10))
        self.dataloader = torch.utils.data.DataLoader(DummyDataset(100), batch_size=10)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experiment = None  # Replace with your Comet.ml Experiment instance

    def test_train(self):
        epoch_loss, epoch_accuracy = train(self.model, self.dataloader, self.criterion, self.optimizer, self.device, self.experiment)
        self.assertIsInstance(epoch_loss, float)
        self.assertIsInstance(epoch_accuracy, float)

    def test_validate(self):
        epoch_loss, epoch_accuracy, true_labels, predicted_labels = validate(self.model, self.dataloader, self.criterion, self.device, self.experiment)
        self.assertIsInstance(epoch_loss, float)
        self.assertIsInstance(epoch_accuracy, float)
        self.assertIsInstance(true_labels, list)
        self.assertIsInstance(predicted_labels, list)

    def test_test(self):
        true_labels, predicted_labels = test(self.model, self.dataloader, self.device)
        self.assertIsInstance(true_labels, list)
        self.assertIsInstance(predicted_labels, list)


if __name__ == "__main__":
    unittest.main()
