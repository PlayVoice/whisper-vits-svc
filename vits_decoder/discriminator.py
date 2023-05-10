import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf
from torch.nn.utils import weight_norm
from .mpd import MultiPeriodDiscriminator
from .mrd import MultiResolutionDiscriminator


class DiscriminatorS(torch.nn.Module):
    def __init__(self):
        super(DiscriminatorS, self).__init__()
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


class Discriminator(nn.Module):
    def __init__(self, hp):
        super(Discriminator, self).__init__()
        self.MRD = MultiResolutionDiscriminator(hp)
        self.MPD = MultiPeriodDiscriminator(hp)
        self.DIS = DiscriminatorS()

    def forward(self, x):
        return self.MRD(x), self.MPD(x), self.DIS(x)


if __name__ == '__main__':
    hp = OmegaConf.load('../config/default.yaml')
    model = Discriminator(hp)

    x = torch.randn(3, 1, 16384)
    print(x.shape)

    mrd_output, mpd_output = model(x)
    for features, score in mpd_output:
        for feat in features:
            print(feat.shape)
        print(score.shape)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

