import os
import numpy as np
import librosa
import torch
import torchcrepe
import argparse


def compute_f0(filename, save, device):
    audio, sr = librosa.load(filename, sr=16000)
    assert sr == 16000
    # Load audio
    audio = torch.tensor(np.copy(audio))[None]
    # Here we'll use a 20 millisecond hop length
    hop_length = 320
    # Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
    # This would be a reasonable range for speech
    fmin = 50
    fmax = 1000
    # Select a model capacity--one of "tiny" or "full"
    model = "full"
    # Pick a batch size that doesn't cause memory errors on your gpu
    batch_size = 512
    # Compute pitch using first gpu
    pitch, periodicity = torchcrepe.predict(
        audio,
        sr,
        hop_length,
        fmin,
        fmax,
        model,
        batch_size=batch_size,
        device=device,
        return_periodicity=True,
    )
    pitch = np.repeat(pitch, 2, -1)  # 320 -> 160 * 2
    periodicity = np.repeat(periodicity, 2, -1)  # 320 -> 160 * 2
    # CREPE was not trained on silent audio. some error on silent need filter.pitPath
    periodicity = torchcrepe.filter.median(periodicity, 9)
    pitch = torchcrepe.filter.mean(pitch, 5)
    pitch[periodicity < 0.1] = 0
    pitch = pitch.squeeze(0)
    np.save(save, pitch, allow_pickle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("-w", "--wav", help="wav", dest="wav")
    parser.add_argument("-p", "--pit", help="pit", dest="pit")
    args = parser.parse_args()
    print(args.wav)
    print(args.pit)
    os.makedirs(args.pit)
    wavPath = args.wav
    pitPath = args.pit

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
            os.makedirs(f"./{pitPath}/{spks}")
            print(f">>>>>>>>>>{spks}<<<<<<<<<<")
            for file in os.listdir(f"./{wavPath}/{spks}"):
                if file.endswith(".wav"):
                    print(file)
                    file = file[:-4]
                    compute_f0(f"{wavPath}/{spks}/{file}.wav", f"{pitPath}/{spks}/{file}.pit", device)
