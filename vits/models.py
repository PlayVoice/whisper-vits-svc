
import torch

from torch import nn
from torch.nn import functional as F
from vits import attentions
from vits import commons
from vits import modules
from vits.utils import f0_to_coarse
from vits_decoder.generator import Generator
from vits.modules_grl import SpeakerClassifier


class TextEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 vec_channels,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2)
        self.hub = nn.Conv1d(vec_channels, hidden_channels, kernel_size=5, padding=2)
        self.pit = nn.Embedding(256, hidden_channels)
        self.enc = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, v, f0):
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        v = torch.transpose(v, 1, -1)  # [b, h, t]
        v = self.hub(v) * x_mask
        x = x + v + self.pit(f0).transpose(1, 2)
        x = self.enc(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask, x


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
            total_logdet = 0
            for flow in self.flows:
                x, log_det = flow(x, x_mask, g=g, reverse=reverse)
                total_logdet += log_det
            return x, total_logdet
        else:
            total_logdet = 0
            for flow in reversed(self.flows):
                x, log_det = flow(x, x_mask, g=g, reverse=reverse)
                total_logdet += log_det
            return x, total_logdet

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
        self.out_channels = out_channels
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
    def __init__(
        self,
        spec_channels,
        segment_size,
        hp
    ):
        super().__init__()
        self.segment_size = segment_size
        self.emb_g = nn.Linear(hp.vits.spk_dim, hp.vits.gin_channels)
        self.enc_p = TextEncoder(
            hp.vits.ppg_dim,
            hp.vits.vec_dim,
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            hp.vits.filter_channels,
            2,
            6,
            3,
            0.1,
        )
        self.speaker_classifier = SpeakerClassifier(
            hp.vits.hidden_channels,
            hp.vits.spk_dim,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            5,
            1,
            16,
            gin_channels=hp.vits.gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            5,
            1,
            4,
            gin_channels=hp.vits.spk_dim
        )
        self.dec = Generator(hp=hp)

    def forward(self, ppg, vec, pit, spec, spk, ppg_l, spec_l):
        ppg = ppg + torch.randn_like(ppg) * 1  # Perturbation
        vec = vec + torch.randn_like(vec) * 2  # Perturbation
        g = self.emb_g(F.normalize(spk)).unsqueeze(-1)
        z_p, m_p, logs_p, ppg_mask, x = self.enc_p(
            ppg, ppg_l, vec, f0=f0_to_coarse(pit))
        z_q, m_q, logs_q, spec_mask = self.enc_q(spec, spec_l, g=g)

        z_slice, pit_slice, ids_slice = commons.rand_slice_segments_with_pitch(
            z_q, pit, spec_l, self.segment_size)
        audio = self.dec(spk, z_slice, pit_slice)

        # SNAC to flow
        z_f, logdet_f = self.flow(z_q, spec_mask, g=spk)
        z_r, logdet_r = self.flow(z_p, spec_mask, g=spk, reverse=True)
        # speaker
        spk_preds = self.speaker_classifier(x)
        return audio, ids_slice, spec_mask, (z_f, z_r, z_p, m_p, logs_p, z_q, m_q, logs_q, logdet_f, logdet_r), spk_preds

    def infer(self, ppg, vec, pit, spk, ppg_l):
        ppg = ppg + torch.randn_like(ppg) * 0.0001  # Perturbation
        z_p, m_p, logs_p, ppg_mask, x = self.enc_p(
            ppg, ppg_l, vec, f0=f0_to_coarse(pit))
        z, _ = self.flow(z_p, ppg_mask, g=spk, reverse=True)
        o = self.dec(spk, z * ppg_mask, f0=pit)
        return o


class SynthesizerInfer(nn.Module):
    def __init__(
        self,
        spec_channels,
        segment_size,
        hp
    ):
        super().__init__()
        self.segment_size = segment_size
        self.enc_p = TextEncoder(
            hp.vits.ppg_dim,
            hp.vits.vec_dim,
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            hp.vits.filter_channels,
            2,
            6,
            3,
            0.1,
        )
        self.flow = ResidualCouplingBlock(
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            5,
            1,
            4,
            gin_channels=hp.vits.spk_dim
        )
        self.dec = Generator(hp=hp)

    def remove_weight_norm(self):
        self.flow.remove_weight_norm()
        self.dec.remove_weight_norm()

    def pitch2source(self, f0):
        return self.dec.pitch2source(f0)

    def source2wav(self, source):
        return self.dec.source2wav(source)

    def inference(self, ppg, vec, pit, spk, ppg_l, source):
        z_p, m_p, logs_p, ppg_mask, x = self.enc_p(
            ppg, ppg_l, vec, f0=f0_to_coarse(pit))
        z, _ = self.flow(z_p, ppg_mask, g=spk, reverse=True)
        o = self.dec.inference(spk, z * ppg_mask, source)
        return o
