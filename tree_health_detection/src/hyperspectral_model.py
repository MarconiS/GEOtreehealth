import torch
import torch.nn as nn
from torchvision.models import resnet50

class HyperspectralModel(nn.Module):
    def __init__(self, num_bands):
        super(HyperspectralModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_bands, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), # Add BatchNorm2d
            nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), # Add BatchNorm2d
            nn.ReLU(inplace=True),)                       
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.avg_pool(x)
        return x.view(x.size(0), -1)