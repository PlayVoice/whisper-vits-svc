
import torch

from torch import nn
from torch.nn import functional as F
from vits import attentions
from vits import commons
from vits import modules

from vits_decoder.generator import Generator
from grad.diffusion import Diffusion
from grad.utils import f0_to_coarse, rand_ids_segments, slice_segments


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
        self.post = Diffusion(hp.vits.inter_channels,
                              64,
                              emb_dim=hp.vits.spk_dim)
        self.dec = Generator(hp=hp)

    def forward(self, ppg, vec, pit, spec, spk, ppg_l, spec_l):
        ppg = ppg + torch.randn_like(ppg) * 1  # Perturbation
        vec = vec + torch.randn_like(vec) * 2  # Perturbation
        g = self.emb_g(F.normalize(spk)).unsqueeze(-1)
        z_p, m_p, logs_p, ppg_mask, x = self.enc_p(
            ppg, ppg_l, vec, f0=f0_to_coarse(pit))
        # wave encoder
        z_q, m_q, logs_q, spec_mask = self.enc_q(spec, spec_l, g=g)
        # SNAC to flow
        z_r, logdet_r = self.flow(z_p, spec_mask, g=spk, reverse=True)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        out_size = self.segment_size
        ids = rand_ids_segments(ppg_l, out_size)
        mel = slice_segments(z_q, ids, out_size)

        mask_y = slice_segments(ppg_mask, ids, out_size)
        mu_y = slice_segments(z_r, ids, out_size)
        # grad
        diff_loss, xt = self.post.compute_loss(spk, mel, mask_y, mu_y)
        return diff_loss

    def infer(self, ppg, vec, pit, spk, ppg_l, n_timesteps=50, temperature=1.0, stoc=False):
        ppg = ppg + torch.randn_like(ppg) * 0.0001  # Perturbation
        z_p, m_p, logs_p, ppg_mask, x = self.enc_p(
            ppg, ppg_l, vec, f0=f0_to_coarse(pit))
        z1, _ = self.flow(z_p, ppg_mask, g=spk, reverse=True)
        # Sample latent representation from terminal distribution N(mu_y, I)
        z2 = z1 + torch.randn_like(z1, device=z1.device) / temperature
        # Generate sample by performing reverse dynamics
        z3 = self.post(spk, z1, ppg_mask, z2, n_timesteps, stoc)
        o = self.dec(spk, z3 * ppg_mask, f0=pit)
        return o
    
    def train_post(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.post.parameters():
            param.requires_grad = True


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
        self.post = Diffusion(hp.vits.inter_channels,
                              64,
                              emb_dim=hp.vits.spk_dim)
        self.dec = Generator(hp=hp)

    def remove_weight_norm(self):
        self.flow.remove_weight_norm()
        self.dec.remove_weight_norm()

    def pitch2source(self, f0):
        return self.dec.pitch2source(f0)

    def source2wav(self, source):
        return self.dec.source2wav(source)

    def inference_no_post(self, ppg, vec, pit, spk, ppg_l, source):
        z_p, m_p, logs_p, ppg_mask, x = self.enc_p(
            ppg, ppg_l, vec, f0=f0_to_coarse(pit))
        z, _ = self.flow(z_p, ppg_mask, g=spk, reverse=True)
        o = self.dec.inference(spk, z * ppg_mask, source)
        return o
    
    def inference(self, ppg, vec, pit, spk, ppg_l, source, n_timesteps=50, temperature=1.0, stoc=False):
        z_p, m_p, logs_p, ppg_mask, x = self.enc_p(
            ppg, ppg_l, vec, f0=f0_to_coarse(pit))
        z1, _ = self.flow(z_p, ppg_mask, g=spk, reverse=True)
        # Sample latent representation from terminal distribution N(mu_y, I)
        z2 = z1 + torch.randn_like(z1, device=z1.device) / temperature
        # Generate sample by performing reverse dynamics
        z3 = self.post(spk, z1, ppg_mask, z2, n_timesteps, stoc)
        o = self.dec.inference(spk, z3 * ppg_mask, source)
        return o
