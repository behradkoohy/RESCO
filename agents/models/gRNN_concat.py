# PyTorch Imports
import torch
import torch.nn as nn
from pfrl.q_functions import DiscreteActionValueHead

class gRNNConc(nn.Module):
    def __init__(self, h, w, obs_space, act_space, k_size, seq_l=10):
        # super(gRNNConc, self).__init__()
        super().__init__()
        # Phase (GRU) Side
        # self.gru = nn.GRU(obs_space[0], seq_l)
        self.gfc1 = nn.Linear(32, 32)
        # Lane Data Side
        # self.conv = nn.Conv2d(64, 64, kernel_size=(k_size, k_size))
        self.conv = nn.Conv2d(obs_space[0], 64, kernel_size=(k_size, k_size))
        self.fc1 = nn.Linear(h * w * 64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, act_space)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
    def forward(self, data):
        # Deal with non sequential data first
        x1 = self.conv(data)
        x1 = self.relu(x1)
        x1 = self.flatten(x1)
        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        x1 = self.relu(x1)

        # x2 = self.gru(deque_sequence, num_layers=2)

        
        # x = torch.cat((x1, x2), dim=1)
        # x = nn.relu(self.fc3(x))
        # x = self.relu(self.fc3(x1))
        x1 = x1.view(x1.size(0), -1)
        return x1
        



