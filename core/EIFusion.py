import torch
import torch.nn as nn
import torch.nn.functional as F


class Fusion(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, 192, 1, padding=0)
        self.conv2 = nn.Conv2d(input_dim, 192, 1, padding=0)
        self.convo = nn.Conv2d(192*2, input_dim, 3, padding=1)

    def forward(self, x1, x2):
        c1 = F.relu(self.conv1(x1))
        c2 = F.relu(self.conv2(x2))
        out = torch.cat([c1, c2], dim=1)
        out = F.relu(self.convo(out))
        return out + x1
