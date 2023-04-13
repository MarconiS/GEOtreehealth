import torch
import torch.nn as nn

class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.identity = torch.eye(k, requires_grad=True).view(-1)

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = nn.AdaptiveMaxPool1d(1)(x)
        x = x.view(batch_size, -1)
        x = torch.relu(self.bn4(self.fc1(x)))
        x = torch.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        init = self.identity.repeat(batch_size, 1)
        matrix = x + init
        matrix = matrix.view(batch_size, -1)
        return matrix

class LiDARModel(nn.Module):
    def __init__(self, num_points=None, num_classes=3):
        super(LiDARModel, self).__init__()
        self.tnet3 = TNet(k=3)
        self.tnet64 = TNet(k=64)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.num_points = num_points

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        input_transform = self.tnet3(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, input_transform)
        x = x.transpose(2, 1)

        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))

        feature_transform = self.tnet64(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, feature_transform)
        x = x.transpose(2, 1)

        x = torch.relu(self.bn3(self.conv3(x)))
        x = nn.AdaptiveMaxPool1d(1)(x)
        x = x.view(batch_size, -1)

        x = torch.relu(self.bn4(self.fc1(x)))
        x = torch.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        return x
