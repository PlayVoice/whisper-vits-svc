import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

class DiscriminatorP(nn.Module):
    def __init__(self, hp, period):
        super(DiscriminatorP, self).__init__()

        self.LRELU_SLOPE = hp.mpd.lReLU_slope
        self.period = period

        kernel_size = hp.mpd.kernel_size
        stride = hp.mpd.stride
        norm_f = weight_norm if hp.mpd.use_spectral_norm == False else spectral_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 64, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            norm_f(nn.Conv2d(64, 128, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            norm_f(nn.Conv2d(128, 256, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            norm_f(nn.Conv2d(256, 512, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), 1, padding=(kernel_size // 2, 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return fmap, x


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, hp):
        super(MultiPeriodDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList(
            [DiscriminatorP(hp, period) for period in hp.mpd.periods]
        )

    def forward(self, x):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x))

        return ret  # [(feat, score), (feat, score), (feat, score), (feat, score), (feat, score)]
