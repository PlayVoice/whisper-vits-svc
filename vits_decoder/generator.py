import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import Conv1d
from torch.nn import ConvTranspose1d
from torch.nn.utils import weight_norm
from torch.nn.utils import remove_weight_norm

from .nsf import SourceModuleHnNSF, TorchSTFT
from .bigv import init_weights, AMPBlock


class SpeakerAdapter(nn.Module):

    def __init__(self,
                 speaker_dim,
                 adapter_dim,
                 epsilon=1e-5
                 ):
        super(SpeakerAdapter, self).__init__()
        self.speaker_dim = speaker_dim
        self.adapter_dim = adapter_dim
        self.epsilon = epsilon
        self.W_scale = nn.Linear(self.speaker_dim, self.adapter_dim)
        self.W_bias = nn.Linear(self.speaker_dim, self.adapter_dim)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.W_scale.weight, 0.0)
        torch.nn.init.constant_(self.W_scale.bias, 1.0)
        torch.nn.init.constant_(self.W_bias.weight, 0.0)
        torch.nn.init.constant_(self.W_bias.bias, 0.0)

    def forward(self, x, speaker_embedding):
        x = x.transpose(1, -1)
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        scale = self.W_scale(speaker_embedding)
        bias = self.W_bias(speaker_embedding)
        y *= scale.unsqueeze(1)
        y += bias.unsqueeze(1)
        y = y.transpose(1, -1)
        return y


class Generator(torch.nn.Module):
    # this is our main BigVGAN model. Applies anti-aliased periodic activation for resblocks.
    def __init__(self, hp):
        super(Generator, self).__init__()
        self.hp = hp
        self.num_kernels = len(hp.gen.resblock_kernel_sizes)
        self.num_upsamples = len(hp.gen.upsample_rates)
        # speaker adaper, 256 should change by what speaker encoder you use
        self.adapter = SpeakerAdapter(hp.vits.spk_dim, hp.gen.upsample_input)
        # pre conv
        self.conv_pre = Conv1d(hp.gen.upsample_input,
                               hp.gen.upsample_initial_channel, 7, 1, padding=3)
        # nsf
        self.f0_upsamp = torch.nn.Upsample(
            scale_factor=int(np.prod(hp.gen.upsample_rates) * hp.gen.istft_hop_size), mode='linear', align_corners=True)
        self.m_source = SourceModuleHnNSF(sampling_rate=hp.data.sampling_rate,
                                          upsample_scale=np.prod(hp.gen.upsample_rates) * hp.gen.istft_hop_size,
                                          harmonic_num=8, voiced_threshod=10)

        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        # transposed conv-based upsamplers.
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(hp.gen.upsample_rates, hp.gen.upsample_kernel_sizes)):
            # print(f'ups: {i} {k}, {u}, {(k - u) // 2}')
            # base
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        hp.gen.upsample_initial_channel // (2 ** i),
                        hp.gen.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2)
                )
            )
            c_cur = hp.gen.upsample_initial_channel // (2 ** (i + 1))
            # nsf
            if i + 1 < len(hp.gen.upsample_rates):
                stride_f0 = np.prod(hp.gen.upsample_rates[i + 1:])
                stride_f0 = int(stride_f0)
                self.noise_convs.append(Conv1d(
                    hp.gen.istft_n_fft + 2, c_cur,
                    kernel_size=stride_f0 * 2, stride=stride_f0, padding=stride_f0 // 2)
                )
                self.noise_res.append(AMPBlock(c_cur, 7, [1, 3, 5]))
            else:
                self.noise_convs.append(Conv1d(hp.gen.istft_n_fft + 2, c_cur, kernel_size=1))
                self.noise_res.append(AMPBlock(c_cur, 11, [1, 3, 5]))

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = hp.gen.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(hp.gen.resblock_kernel_sizes, hp.gen.resblock_dilation_sizes):
                self.resblocks.append(AMPBlock(ch, k, d))

        # post conv
        self.post_n_fft = hp.gen.istft_n_fft
        self.conv_post = weight_norm(Conv1d(ch, self.post_n_fft + 2, 7, 1, padding=3))
        # weight initialization
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.stft = TorchSTFT(filter_length=hp.gen.istft_n_fft,
                              hop_length=hp.gen.istft_hop_size, win_length=hp.gen.istft_n_fft)

    def forward(self, spk, x, f0):
        # Perturbation
        x = x + torch.randn_like(x)
        # adapter
        x = self.adapter(x, spk)
        x = self.conv_pre(x)
        # nsf
        f0 = f0[:, None]
        f0 = self.f0_upsamp(f0).transpose(1, 2)
        har_source = self.m_source(f0)
        har_source = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = self.stft.transform(har_source)
        har = torch.cat([har_spec, har_phase], dim=1)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            # upsampling
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            # nsf
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source)
            x = x + x_source
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # post conv
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:, :self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
        x = self.stft.inverse(spec, phase)
        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()

    def eval(self, inference=False):
        super(Generator, self).eval()
        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def pitch2source(self, f0):
        f0 = f0[:, None]
        f0 = self.f0_upsamp(f0).transpose(1, 2)  # [1,len,1]
        har_source = self.m_source(f0)
        har_source = har_source.transpose(1, 2)  # [1,1,len]
        return har_source

    def source2wav(self, audio):
        MAX_WAV_VALUE = 32768.0
        audio = audio.squeeze()
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
        audio = audio.short()
        return audio.cpu().detach().numpy()

    def inference(self, spk, x, har_source):
        # Perturbation
        x = x + torch.randn_like(x) * 0.1
        # adapter
        x = self.adapter(x, spk)
        x = self.conv_pre(x)
        # nsf
        har_source = har_source.squeeze(1)
        har_spec, har_phase = self.stft.transform(har_source)
        har = torch.cat([har_spec, har_phase], dim=1)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            # upsampling
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            # nsf
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source)
            x = x + x_source
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # post conv
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:, :self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
        x = self.stft.inverse(spec, phase)
        return x
