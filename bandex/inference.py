import os
import argparse
import librosa
import torch
from scipy.io.wavfile import write
import numpy as np

SCALE = 3


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(os.path.join(
        "bandex", "hifi-gan-bwe-vctk-48kHz.pt")).to(device)
    x, sr = librosa.load(args.wave, sr=16000)

    hop_size = 320  # whisper hop size
    hop_count = len(x) // hop_size
    hop_frame = 10
    bwe_chunk = 500  # 10S
    bwe_index = 0
    bwe_audio = []

    while (bwe_index + bwe_chunk < hop_count):
        if (bwe_index == 0):  # start frame
            cut_s_16k = 0
            cut_s_48k = 0
        else:
            cut_s_16k = (bwe_index - hop_frame) * hop_size
            cut_s_48k = hop_frame * hop_size * SCALE

        if (bwe_index + bwe_chunk + hop_frame > hop_count):  # end frame
            cut_e_16k = (bwe_index + bwe_chunk) * hop_size
            cut_e_48k = 0
        else:
            cut_e_16k = (bwe_index + bwe_chunk + hop_frame) * hop_size
            cut_e_48k = -1 * hop_frame * hop_size * SCALE
        x_chunk = x[cut_s_16k:cut_e_16k]

        with torch.no_grad():
            i_audio = torch.from_numpy(x_chunk).to(device)
            o_audio = model(i_audio, sr).data.cpu().float().numpy()
        o_audio = o_audio[cut_s_48k:cut_e_48k]
        bwe_audio.extend(o_audio)
        bwe_index = bwe_index + bwe_chunk

    if (bwe_index < hop_count):
        cut_s_16k = bwe_index - hop_frame
        cut_s_48k = hop_frame * hop_size * SCALE
        x_chunk = x[cut_s_16k * hop_size:]
        with torch.no_grad():
            i_audio = torch.from_numpy(x_chunk).to(device)
            o_audio = model(i_audio, sr).data.cpu().float().numpy()
        o_audio = o_audio[cut_s_48k:]
        bwe_audio.extend(o_audio)
    bwe_audio = np.asarray(bwe_audio)
    write("svc_out_48k.wav", 48000, bwe_audio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--wave', type=str, required=True,
                        help="Path of raw audio.")
    args = parser.parse_args()
    main(args)
