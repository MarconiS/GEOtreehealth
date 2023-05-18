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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import timm
import torch.nn as nn

import torch
from torch.nn import Linear, ReLU, Dropout
from torch.nn.functional import log_softmax, relu
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
        self.lidar_branch = DGCNN(in_channels, lidar_out_features)

        # Define the fully connected layer
        num_features = hsi_out_features + rgb_out_features + lidar_out_features
        self.fc = nn.Linear(num_features, num_classes)  # replace num_classes with the number of your classes

    def forward(self, hsi, rgb, lidar):
        # Pass inputs through corresponding branches
        hsi_out = self.hsi_branch(hsi)
        rgb_out = self.rgb_branch(rgb)
        lidar_out = self.lidar_branch(lidar)

        # Use adaptive average pooling to handle different image sizes
        hsi_out = nn.AdaptiveAvgPool2d((1,1))(hsi_out)
        #lidar_out = torch.max(lidar_out, -1, keepdim=False)[0]  # apply global max pooling to lidar_out

        
        # Flatten the outputs
        hsi_out = hsi_out.view(hsi_out.size(0), -1)
        lidar_out = lidar_out.view(lidar_out.size(0), -1)
        
        # Concatenate the outputs   
        out = torch.cat((hsi_out, rgb_out, lidar_out), dim=1)

        # Pass through final layer to make prediction
        out = self.fc(out)

        return out


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
        x = x[0]  # unpack the tuple

        # Apply spatial attention to the output
        sa = self.spatial_attention(x)

        # Multiply the feature map by the attention map
        x = x * sa

        return x


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1] 
    return idx



class TransformNet(nn.Module):
    def __init__(self, in_channels=6):
        super(TransformNet, self).__init__()
        self.in_channels = in_channels  # define in_channels
        self.conv1 = torch.nn.Conv1d(self.in_channels, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.in_channels*self.in_channels)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        batch_size = x.size(0)

        # Transpose the input tensor here
        x = x.transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.in_channels).flatten().astype(np.float32))).view(1, self.in_channels*self.in_channels).repeat(batch_size, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.in_channels, self.in_channels)
        return x



class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super(EdgeConv, self).__init__()
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def get_graph_feature(self, x):
        idx = knn(x, k=self.k)
        device = torch.device('cuda')
        idx_base = torch.arange(0, self.batch_size, device=device).view(-1, 1, 1)*self.num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        x = x.transpose(2, 1).contiguous()
        feature = x.view(self.batch_size*self.num_points, -1)[idx, :]
        feature = feature.view(self.batch_size, self.num_points, self.k, self.in_channels) 
        x = x.view(self.batch_size, self.num_points, 1, self.in_channels).repeat(1, 1, self.k, 1)
        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        return feature

    def forward(self, x):
        self.batch_size = x.size(0)
        self.num_points = x.size(2)
        x = self.get_graph_feature(x)
        x = self.conv(x)
        x = x.max(dim=-1, keepdim=False)[0]
        return x
    

class DGCNN(nn.Module):
    def __init__(self, in_channels, lidar_out_features, k=20):
        super(DGCNN, self).__init__()
        self.in_channels = in_channels
        self.lidar_out_features = lidar_out_features
        self.k = k
        self.transform_net = TransformNet(in_channels)
        self.edgeconv1 = EdgeConv(in_channels, 64, k)
        self.edgeconv2 = EdgeConv(64, 128, k)
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, lidar_out_features)

    # x = lidar.to(device) 
    def forward(self, x):
        # permute the dimensions to match the expected input shape
        #
        t = self.transform_net(x)
        x = torch.bmm(x, t)
        x = x.permute(0, 2, 1) # permute dimensions to be [batch_size, num_channels, num_points]
        x = self.edgeconv1(x)
        x = self.edgeconv2(x)
        x = x.max(dim=2, keepdim=True)[0] # global max pooling
        x = x.squeeze(-1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Then you can include DGCNN in your MultiModalNet as self.lidar_branch
