import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

class QNetwork(nn.Module):
    def __init__(self, action_size, seed):
        super(QNetwork, self).__init__()
        nfilters = [128, 128*2, 128*2]
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv3d(3, nfilters[0], kernel_size=(1, 3, 3), stride=(1,3,3))
        self.bn1 = nn.BatchNorm3d(nfilters[0])
        self.conv2 = nn.Conv3d(nfilters[0], nfilters[1], kernel_size=(1, 3, 3), stride=(1,3,3))
        self.bn2 = nn.BatchNorm3d(nfilters[1])
        self.conv3 = nn.Conv3d(nfilters[1], nfilters[2], kernel_size=(4, 3, 3), stride=(1,3,3))
        self.bn3 = nn.BatchNorm3d(nfilters[2])
        fc = [2304, 1024]
        self.fc1 = nn.Linear(fc[0], fc[1])
        self.fc2 = nn.Linear(fc[1], action_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x