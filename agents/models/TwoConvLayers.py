import torch
import torch.nn as nn


class TwoConvLayer(nn.Module):
    def __init__(self, h, w, obs_space, act_space, k_size, seq_l=10):

        super().__init__()

        self.gfc1 = nn.Linear(32, 32)

        self.conv = nn.Conv2d(obs_space[0], 64, kernel_size=(k_size, k_size))
        self.conv2 = nn.Conv2d(64, 64, 4)
        self.fc1 = nn.Linear(h * w * 64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, act_space)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, data):
        # Deal with non sequential data first
        x1 = self.conv(data)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.relu(x1)
        x1 = self.flatten(x1)
        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        x1 = self.relu(x1)

        x1 = x1.view(x1.size(0), -1)
        return x1


