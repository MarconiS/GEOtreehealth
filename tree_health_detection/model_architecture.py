import torch.nn as nn
import torchvision.models as models
from torch.nn import AdaptiveAvgPool2d
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTForImageClassification
import numpy as np

import timm
import torch.nn as nn

import torch
from torch.nn import Sequential as Seq, Linear, ReLU
import torch.nn.functional as F
#from torch_geometric.nn import DynamicEdgeConv, global_max_pool

class MultiModalNet(nn.Module):
    def __init__(self, in_channels, num_classes, num_bands,
                 hsi_out_features, rgb_out_features, lidar_out_features):
        super(MultiModalNet, self).__init__()
        self.num_bands= num_bands
        self.hsi_out_features = hsi_out_features
        self.rgb_out_features = rgb_out_features
        self.lidar_out_features = lidar_out_features
        self.num_classes = num_classes
        # 1. HSI branch: Spectral Attention Network
        self.hsi_branch = SpectralAttentionNetwork(num_bands, hsi_out_features) # to be implemented
        
        # 2. RGB branch: Spatial Attention + Hybrid Visual Transformer
        self.rgb_branch = HybridViTWithAttention(num_classes=rgb_out_features)

        # 3. LiDAR branch: DGCNN
        #self.lidar_branch = DGCNN(in_channels, lidar_out_features) # to be implemented
        
        # Define the fully connected layer
        num_features = hsi_out_features + rgb_out_features# + lidar_out_features
        self.fc = nn.Linear(num_features, num_classes)  # replace num_classes with the number of your classes

    def forward(self, hsi, rgb, lidar):
        # Pass inputs through corresponding branches
        hsi_out = self.hsi_branch(hsi)
        rgb_out = self.rgb_branch(rgb)
        #lidar_out = self.lidar_branch(lidar)

        # Use adaptive average pooling to handle different image sizes
        hsi_out = nn.AdaptiveAvgPool2d((1,1))(hsi_out)
        
        # Flatten the outputs
        hsi_out = hsi_out.view(hsi_out.size(0), -1)
        #lidar_out = lidar_out.view(lidar_out.size(0), -1)
        
        # Concatenate the outputs   ,lidar_out
        out = torch.cat((hsi_out, rgb_out), dim=1)

        # Pass through final layer to make prediction
        out = self.fc(out)

        return out


import torch
from torch.nn import Linear, ReLU, Dropout
from torch.nn.functional import log_softmax, relu


class SpectralAttentionLayer(nn.Module):
    def __init__(self, channel):
        super(SpectralAttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpectralAttentionNetwork(nn.Module):
    def __init__(self, num_bands, hsi_out_features):
        super(SpectralAttentionNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=num_bands, out_channels=hsi_out_features, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hsi_out_features)
        self.relu = nn.ReLU(inplace=True)

        self.spectral_attention = SpectralAttentionLayer(hsi_out_features)

    def forward(self, x):
        # Perform initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Apply spectral attention
        sa = self.spectral_attention(x)

        # Multiply the feature map by the attention map
        x = x * sa

        return x


class VisualTransformer(nn.Module):
    def __init__(self, num_classes=3):
        super(VisualTransformer, self).__init__()

        # Load the ViT base 16 with an image size of 224
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)

        # Change the number of output features in case you have a different number of classes
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        x = self.vit.get_intermediate_layers(x)
        print(x)
        return x

class HybridViTWithAttention(nn.Module):
    def __init__(self, num_classes=3):
        super(HybridViTWithAttention, self).__init__()

        # Load a pre-trained CNN for feature extraction
        self.feature_extractor = timm.create_model('resnet50', pretrained=True)

        # Remove the final layers of the CNN
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])

        # Initialize the spatial attention module
        self.spatial_attention = SpatialAttention()

        # Add a global average pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Add a fully connected layer for the final prediction
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Extract features using the CNN
        features = self.feature_extractor(x)

        # Apply the spatial attention mechanism to the features
        sa = self.spatial_attention(features)

        # Multiply the feature map by the attention map
        attended_features = features * sa

        # Apply global average pooling to the attended features
        pooled_features = self.global_avg_pool(attended_features)

        # Flatten the pooled features
        flattened_features = pooled_features.view(pooled_features.size(0), -1)

        # Pass the flattened features through the fully connected layer
        x = self.fc(flattened_features)

        return x
    



class DGCNN(torch.nn.Module):
    def __init__(self, k=20, output_channels=10):
        super(DGCNN, self).__init__()
        self.k = k
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(12, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 128))
        self.nn2 = torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 256))
        self.lin1 = torch.nn.Linear(256, 256)
        self.lin2 = torch.nn.Linear(256, output_channels)

    def forward(self, x, batch):
        x = self.edge_conv(x, self.nn1)
        x = self.edge_conv(x, self.nn2)
        x = self.global_max_pool(x, batch)
        x = torch.relu(self.lin1(x))
        x = torch.nn.Dropout(p=0.5)(x)
        x = self.lin2(x)
        return torch.log_softmax(x, dim=-1)
    
    def edge_conv(self, x, nn):
        row, col = torch.topk(x, self.k, dim=-1)
        row, col = row.long(), col.long()  # explicitly cast to long
        x = torch.cat([x[col], x[row]], dim=-1)
        x = nn(x)
        x, _ = torch.max(x, dim=-1)
        return x


    def global_max_pool(self, x, batch):
        batch_size = torch.max(batch) + 1
        index = torch.arange(batch.size(0), device=batch.device)
        index = index + batch * batch_size
        pooled = torch_scatter.scatter_max(x, index, dim=0)[0]
        return pooled


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SpatialAttentionResnetTransformer(nn.Module):
    def __init__(self):
        super(SpatialAttentionResnetTransformer, self).__init__()

        self.vit = VisualTransformer(num_classes=3)  # this may effectively be something differnet than self.num_classes
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # Pass input through Visual Transformer
        x = self.vit(x)
        print(x)
        x = x[0]  # unpack the tuple
        print(x.shape)

        # Apply spatial attention to the output
        sa = self.spatial_attention(x)

        # Multiply the feature map by the attention map
        x = x * sa

        return x
