import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import librosa
import argparse
import numpy as np
from scipy.io.wavfile import write
from vad.utils import init_jit_model, get_speech_timestamps


def load_audio(file: str, sr: int = 16000):
    x, sr = librosa.load(file, sr=sr)
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ref', type=str, required=True,
                        help="Path of ref audio.")
    parser.add_argument('--svc', type=str, required=True,
                        help="Path of svc audio.")
    parser.add_argument('--out', type=str, required=True,
                        help="Path of out audio.")

    args = parser.parse_args()
    print("svc in wave :", args.ref)
    print("svc out wave :", args.svc)
    print("svc post wave :", args.out)

    model = init_jit_model(os.path.join('vad/assets', 'silero_vad.jit'))
    model.eval()

    ref_wave = load_audio(args.ref, sr=16000)
    tmp_wave = torch.from_numpy(ref_wave).squeeze(0)
    tag_wave = get_speech_timestamps(
        tmp_wave, model, threshold=0.2, sampling_rate=16000)

    ref_wave[:] = 0
    for tag in tag_wave:
        ref_wave[tag["start"]:tag["end"]] = 1

    ref_wave = np.repeat(ref_wave, 2, -1)
    svc_wave = load_audio(args.svc, sr=32000)

    min_len = min(len(ref_wave), len(svc_wave))
    ref_wave = ref_wave[:min_len]
    svc_wave = svc_wave[:min_len]
    svc_wave[ref_wave == 0] = 0

    write(args.out, 32000, svc_wave)
