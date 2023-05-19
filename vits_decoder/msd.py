import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class ScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(ScaleDiscriminator, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(1, 16, 15, 1, padding=7)),
            weight_norm(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            weight_norm(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return [(fmap, x)]
