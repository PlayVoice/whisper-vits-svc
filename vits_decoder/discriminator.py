import torch
import torch.nn as nn

from .mpd import MultiPeriodDiscriminator
from .mrd import MultiResolutionDiscriminator
from omegaconf import OmegaConf

class Discriminator(nn.Module):
    def __init__(self, hp):
        super(Discriminator, self).__init__()
        self.MRD = MultiResolutionDiscriminator(hp)
        self.MPD = MultiPeriodDiscriminator(hp)

    def forward(self, x):
        return self.MRD(x), self.MPD(x)

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

