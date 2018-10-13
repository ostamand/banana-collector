import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

class QNetwork(nn.Module):
    def __init__(self, state_channels, action_size, seed):
        """
        
        """
        super(QNetwork, self).__init__()
        torch.manual_seed(seed)
        
        self.state_channels = state_channels
        self.action_size = action_size
        self.seed = seed
        
        # 32 8x8 filters with stride 4
        self.conv1 = nn.Conv2d(state_channels, 32, 8, stride=4)
        # 64 4x4 filters with stride 2 
        self.conv2 = nn.Conv2d(32,64,4, stride=2)
        # 64 3x3 filters with stride 1 
        self.conv3 = nn.Conv2d(64,64,3, stride=1)
        # 64x7x7, 1024
        self.fc1 = nn.Linear(64*7*7, 1024)
        # 1024, action_size
        self.fc2 = nn.Linear(1024, action_size)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.shape[0], -1)))
        x = self.fc2(x)
        return x 