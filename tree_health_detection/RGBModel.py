import torch
import torch.nn as nn
from torchvision.models import resnet50

class RGBModel(nn.Module):
    def __init__(self):
        super(RGBModel, self).__init__()
        base_model = resnet50(pretrained=True)
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        return x.view(x.size(0), -1)