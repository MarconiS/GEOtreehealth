import torch
import torch.nn as nn
from torchvision.models import resnet50




class MultiModalModel(nn.Module):
    def __init__(self, num_classes, num_bands):
        super(MultiModalModel, self).__init__()
        self.rgb_model = tree_health_detection.RGBModel()
        self.hyperspectral_model = tree_health_detection.HyperspectralModel(num_bands)
        self.lidar_model = tree_health_detection.LiDARModel()
        #2048 are the features from the RGB model, 64 are the features from the hyperspectral model
        self.fc1 = nn.Linear(2048 + 64 + 3, 512) #+ lidar_output_size
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, rgb, hs, lidar):
        rgb_features = self.rgb_model(rgb)
        hs_features = self.hyperspectral_model(hs)
        lidar_features = self.lidar_model(lidar)
        combined_features = torch.cat((rgb_features, hs_features, lidar_features), dim=1)
        x = torch.relu(self.fc1(combined_features))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

