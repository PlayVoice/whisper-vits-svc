import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import Conv1d
from torch.nn import ConvTranspose1d
from torch.nn.utils import weight_norm
from torch.nn.utils import remove_weight_norm

from .nsf import SourceModuleHnNSF
from .bigv import init_weights, AMPBlock, SnakeAlias


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
            scale_factor=np.prod(hp.gen.upsample_rates))
        self.m_source = SourceModuleHnNSF(sampling_rate=hp.data.sampling_rate)
        self.noise_convs = nn.ModuleList()
        # transposed conv-based upsamplers. does not apply anti-aliasing
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
            # nsf
            if i + 1 < len(hp.gen.upsample_rates):
                stride_f0 = np.prod(hp.gen.upsample_rates[i + 1:])
                stride_f0 = int(stride_f0)
                self.noise_convs.append(
                    Conv1d(
                        1,
                        hp.gen.upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(
                    Conv1d(1, hp.gen.upsample_initial_channel //
                           (2 ** (i + 1)), kernel_size=1)
                )

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = hp.gen.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(hp.gen.resblock_kernel_sizes, hp.gen.resblock_dilation_sizes):
                self.resblocks.append(AMPBlock(ch, k, d))

        # post conv
        self.activation_post = SnakeAlias(ch)
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        # weight initialization
        self.ups.apply(init_weights)

    def forward(self, spk, x, f0):
        # Perturbation
        x = x + torch.randn_like(x)
        # adapter
        x = self.adapter(x, spk)
        x = self.conv_pre(x)
        x = x * torch.tanh(F.softplus(x))
        # nsf
        f0 = f0[:, None]
        f0 = self.f0_upsamp(f0).transpose(1, 2)
        har_source = self.m_source(f0)
        har_source = har_source.transpose(1, 2)

        for i in range(self.num_upsamples):
            # upsampling
            x = self.ups[i](x)
            # nsf
            x_source = self.noise_convs[i](har_source)
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
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
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
        # adapter
        x = self.adapter(x, spk)
        x = self.conv_pre(x)
        x = x * torch.tanh(F.softplus(x))

        for i in range(self.num_upsamples):
            # upsampling
            x = self.ups[i](x)
            # nsf
            x_source = self.noise_convs[i](har_source)
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
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x
