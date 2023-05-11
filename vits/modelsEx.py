
import torch

from torch import nn
from torch.nn import functional as F
from vits import commons
from vits.utils import f0_to_coarse
from vits_decoder.generator import Generator
from vits.modules_grl import SpeakerClassifier
from vits.models import TextEncoder, ResidualCouplingBlock, PosteriorEncoder


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

    def forward(self, ppg, pit, spec, spk, ppg_l, spec_l):
        ppg = ppg + torch.randn_like(ppg)  # Perturbation
        g = self.emb_g(F.normalize(spk)).unsqueeze(-1)
        z_p, m_p, logs_p, ppg_mask, x = self.enc_p(
            ppg, ppg_l, f0=f0_to_coarse(pit))
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

    def infer(self, ppg, pit, spk, ppg_l):
        z_p, m_p, logs_p, ppg_mask, x = self.enc_p(
            ppg, ppg_l, f0=f0_to_coarse(pit))
        z, _ = self.flow(z_p, ppg_mask, g=spk, reverse=True)
        o = self.dec(spk, z * ppg_mask, f0=pit)
        return o

    def extend_train(self):
        for p in self.enc_q.parameters():
           p.requires_grad = False
