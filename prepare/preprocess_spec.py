import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
import torch
import argparse

from vits import spectrogram
from vits import utils
from omegaconf import OmegaConf


def compute_spec(hps, filename, specname):
    audio, sampling_rate = utils.load_wav_to_torch(filename)
    assert sampling_rate == hps.sampling_rate, f"{sampling_rate} is not {hps.sampling_rate}"
    audio_norm = audio / hps.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    n_fft = hps.filter_length
    sampling_rate = hps.sampling_rate
    hop_size = hps.hop_length
    win_size = hps.win_length
    spec = spectrogram.spectrogram_torch(
        audio_norm, n_fft, sampling_rate, hop_size, win_size, center=False)
    spec = torch.squeeze(spec, 0)
    torch.save(spec, specname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("-w", "--wav", help="wav", dest="wav")
    parser.add_argument("-s", "--spe", help="spe", dest="spe")
    args = parser.parse_args()
    print(args.wav)
    print(args.spe)
    os.makedirs(args.spe)
    wavPath = args.wav
    spePath = args.spe
    hps = OmegaConf.load("./configs/base.yaml")

    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
            os.makedirs(f"./{spePath}/{spks}")
            print(f">>>>>>>>>>{spks}<<<<<<<<<<")
            for file in os.listdir(f"./{wavPath}/{spks}"):
                if file.endswith(".wav"):
                    # print(file)
                    file = file[:-4]
                    compute_spec(hps.data, f"{wavPath}/{spks}/{file}.wav", f"{spePath}/{spks}/{file}.pt")
 