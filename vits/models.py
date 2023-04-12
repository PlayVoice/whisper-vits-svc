import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

import attentions
import commons
import modules

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding
from vdecoder.hifigan.models import Generator
from utils import f0_to_coarse


class TextEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0,
                 filter_channels=None,
                 n_heads=None,
                 p_dropout=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        self.f0_emb = nn.Embedding(256, hidden_channels)

        self.enc_ = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)

    def forward(self, x, x_lengths, f0=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(
            x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = x + self.f0_emb(f0).transpose(1, 2)
        x = self.enc_(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask

        return z, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

    def remove_weight_norm(self):
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels,
        ssl_dim,
        n_speakers,
        **kwargs
    ):

        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.spk_embed_dim = 256
        self.ssl_dim = ssl_dim
        self.emb_g = nn.Linear(self.spk_embed_dim, gin_channels)

        self.enc_p_ = TextEncoder(ssl_dim, inter_channels, hidden_channels,
                                  5, 1, 16, 0, filter_channels, n_heads, p_dropout)
        hps = {
            "sampling_rate": 48000,
            "inter_channels": 192,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [10, 8, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "gin_channels": 256,
        }
        self.dec = Generator(h=hps)
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels
        )
        self.emb_g = nn.Linear(self.spk_embed_dim, gin_channels)

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    def forward(self, c, f0, spec, g=None, mel=None, c_lengths=None, spec_lengths=None):
        if c_lengths == None:
            c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
        if spec_lengths == None:
            spec_lengths = (torch.ones(spec.size(0)) *
                            spec.size(-1)).to(spec.device)

        g = self.emb_g(g).transpose(1, 2)

        z_ptemp, m_p, logs_p, _ = self.enc_p_(
            c, c_lengths, f0=f0_to_coarse(f0))
        z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)

        z_p = self.flow(z, spec_mask, g=g)
        z_slice, pitch_slice, ids_slice = commons.rand_slice_segments_with_pitch(
            z, f0, spec_lengths, self.segment_size)

        # o = self.dec(z_slice, g=g)
        o = self.dec(z_slice, g=g, f0=pitch_slice)

        return o, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, c, f0, g=None, mel=None, c_lengths=None):
        if c_lengths == None:
            c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
        g = self.emb_g(g).transpose(1, 2)

        z_p, m_p, logs_p, c_mask = self.enc_p_(
            c, c_lengths, f0=f0_to_coarse(f0))
        z = self.flow(z_p, c_mask, g=g, reverse=True)

        o = self.dec(z * c_mask, g=g, f0=f0)

        return o
