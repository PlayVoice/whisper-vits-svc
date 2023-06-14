import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
import numpy as np
import argparse
import torch

from whisper.model import Whisper, ModelDimensions
from whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram


def load_model(path) -> Whisper:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location=device)
    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"])
    del model.decoder
    model.eval()
    model.half()
    model.to(device)
    return model


def pred_ppg(whisper: Whisper, wavPath, ppgPath):
    audio = load_audio(wavPath)
    audln = audio.shape[0]
    ppg_a = []
    idx_s = 0
    while (idx_s + 25 * 16000 < audln):
        short = audio[idx_s:idx_s + 25 * 16000]
        idx_s = idx_s + 25 * 16000
        ppgln = 25 * 16000 // 320
        # short = pad_or_trim(short)
        mel = log_mel_spectrogram(short).half().to(whisper.device)
        with torch.no_grad():
            ppg = whisper.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
            ppg = ppg[:ppgln,]  # [length, dim=1024]
            ppg_a.extend(ppg)
    if (idx_s < audln):
        short = audio[idx_s:audln]
        ppgln = (audln - idx_s) // 320
        # short = pad_or_trim(short)
        mel = log_mel_spectrogram(short).half().to(whisper.device)
        with torch.no_grad():
            ppg = whisper.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
            ppg = ppg[:ppgln,]  # [length, dim=1024]
            ppg_a.extend(ppg)
    np.save(ppgPath, ppg_a, allow_pickle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("-w", "--wav", help="wav", dest="wav")
    parser.add_argument("-p", "--ppg", help="ppg", dest="ppg")
    args = parser.parse_args()
    print(args.wav)
    print(args.ppg)

    wavPath = args.wav
    ppgPath = args.ppg

    whisper = load_model(os.path.join("whisper_pretrain", "medium.pt"))
    pred_ppg(whisper, wavPath, ppgPath)
