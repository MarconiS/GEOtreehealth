import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SpectralAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpectralAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        attn = F.softmax(self.conv1(x), dim=1)
        attn = torch.mean(attn, dim=(2, 3), keepdim=True)  # Spatial average
        return attn * x
    

class SpectralAttentionClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SpectralAttentionClassifier, self).__init__()
        self.spectral_attention = SpectralAttention(in_channels, 1)
        self.fc1 = nn.Linear(in_channels, 256)
        self.fc2 = nn.Linear(256, 128)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Add this layer
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.spectral_attention(x)
        x = x.flatten(2).permute(0, 2, 1)  # Flatten spatial dimensions and permute
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.avg_pool(x)  # Apply the adaptive average pooling
        x = x.squeeze(-1)  # Remove the last singleton dimension
        x = self.fc3(x.view(-1, 128))  # Reshape the input tensor for the last fully connected layer
        return x

