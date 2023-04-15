import os
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
    return model.to(device)


def pred_ppg(whisper: Whisper, wavPath, ppgPath):
    audio = load_audio(wavPath)
    audln = audio.shape[0]
    ppgln = audln // 320
    # audio = pad_or_trim(audio)
    mel = log_mel_spectrogram(audio).to(whisper.device)
    with torch.no_grad():
        ppg = whisper.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
        ppg = ppg[:ppgln,] # [length, dim=1024]
        print(ppg.shape)
        np.save(ppgPath, ppg, allow_pickle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("-w", "--wav", help="wav", dest="wav")
    parser.add_argument("-p", "--ppg", help="ppg", dest="ppg")
    args = parser.parse_args()
    print(args.wav)
    print(args.ppg)
    os.makedirs(args.ppg)
    wavPath = args.wav
    ppgPath = args.ppg

    whisper = load_model(os.path.join("whisper_pretrain", "medium.pt"))

    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
            os.makedirs(f"./{ppgPath}/{spks}")
            print(f">>>>>>>>>>{spks}<<<<<<<<<<")
            for file in os.listdir(f"./{wavPath}/{spks}"):
                if file.endswith(".wav"):
                    # print(file)
                    file = file[:-4]
                    pred_ppg(whisper, f"{wavPath}/{spks}/{file}.wav", f"{ppgPath}/{spks}/{file}.ppg")
        else:
            file = spks
            if file.endswith(".wav"):
                # print(file)
                file = file[:-4]
                pred_ppg(whisper, f"{wavPath}/{file}.wav", f"{ppgPath}/{file}.ppg")
