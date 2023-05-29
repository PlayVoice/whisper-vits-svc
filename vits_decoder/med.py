import torch
import torchaudio
import typing as T


class MelspecDiscriminator(torch.nn.Module):
    """mel spectrogram (frequency domain) discriminator"""

    def __init__(self) -> None:
        super().__init__()
        self.SAMPLE_RATE = 48000
        # mel filterbank transform
        self._melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.SAMPLE_RATE,
            n_fft=2048,
            win_length=int(0.025 * self.SAMPLE_RATE),
            hop_length=int(0.010 * self.SAMPLE_RATE),
            n_mels=128,
            power=1,
        )

        # time-frequency 2D convolutions
        kernel_sizes = [(7, 7), (4, 4), (4, 4), (4, 4)]
        strides = [(1, 2), (1, 2), (1, 2), (1, 2)]
        self._convs = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=1 if i == 0 else 32,
                        out_channels=64,
                        kernel_size=k,
                        stride=s,
                        padding=(1, 2),
                        bias=False,
                    ),
                    torch.nn.BatchNorm2d(num_features=64),
                    torch.nn.GLU(dim=1),
                )
                for i, (k, s) in enumerate(zip(kernel_sizes, strides))
            ]
        )

        # output adversarial projection
        self._postnet = torch.nn.Conv2d(
            in_channels=32,
            out_channels=1,
            kernel_size=(15, 3),
            stride=(1, 2),
        )

    def forward(self, x: torch.Tensor) -> T.Tuple[torch.Tensor, T.List[torch.Tensor]]:
        # apply the log-scale mel spectrogram transform
        x = torch.log(self._melspec(x) + 1e-5)

        # compute hidden layers and feature maps
        f = []
        for c in self._convs:
            x = c(x)
            f.append(x)

        # apply the output projection and global average pooling
        x = self._postnet(x)
        x = x.mean(dim=[-2, -1])

        return [(f, x)]
