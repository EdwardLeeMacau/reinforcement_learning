"""
model.py

File to draft the model architecture for 2048-v0
"""

import torch
import torch.nn as nn
from torch import Tensor

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        # 1x2 conv, 2x1 conv
        self.conv1_1 = nn.Conv2d(16, 128, kernel_size=(1, 2), stride=(1, 1))
        self.conv1_2 = nn.Conv2d(16, 128, kernel_size=(2, 1), stride=(1, 1))

        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=(1, 2), stride=(1, 1))
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=(2, 1), stride=(1, 1))

        self.activation = nn.ReLU()

    def forward(self, observations: Tensor) -> Tensor:
        # calculate 1x2 conv and 2x1 conv
        x1: Tensor = self.conv1_1(observations)
        x2: Tensor = self.conv1_2(observations)

        # Add activation
        x1 = self.activation(x1)
        x2 = self.activation(x2)

        # calculate 1x2 conv and 2x1 conv again
        x1_1: Tensor = self.conv2_1(x1)
        x1_2: Tensor = self.conv2_2(x1)
        x2_1: Tensor = self.conv2_1(x2)
        x2_2: Tensor = self.conv2_2(x2)

        # Add activation
        x1_1 = self.activation(x1_1)
        x1_2 = self.activation(x1_2)
        x2_1 = self.activation(x2_1)
        x2_2 = self.activation(x2_2)

        # Flatten and concat
        x1 = x1.flatten(start_dim=1)
        x2 = x2.flatten(start_dim=1)
        x1_1 = x1_1.flatten(start_dim=1)
        x1_2 = x1_2.flatten(start_dim=1)
        x2_1 = x2_1.flatten(start_dim=1)
        x2_2 = x2_2.flatten(start_dim=1)

        # torch.Size([1, 7424])
        return torch.cat((x1, x2, x1_1, x1_2, x2_1, x2_2), dim=1)

if __name__ == "__main__":
    model = FeatureExtractor()

    observations = torch.zeros(1, 16, 4, 4)
    print(model(observations).shape)
