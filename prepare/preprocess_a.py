import os
import librosa
import argparse
import numpy as np

from scipy.io import wavfile


def resample_wave(wav_in, wav_out, sample_rate):
    wav, _ = librosa.load(wav_in, sr=sample_rate)
    wav = wav / np.abs(wav).max() * 0.6
    wav = wav / max(0.01, np.max(np.abs(wav))) * 32767 * 0.6
    wavfile.write(wav_out, sample_rate, wav.astype(np.int16))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("-w", "--wav", help="wav", dest="wav")
    parser.add_argument("-o", "--out", help="out", dest="out")
    parser.add_argument("-s", "--sr", help="sample rate", dest="sr", type=int)
    args = parser.parse_args()
    print(args.wav)
    print(args.out)
    print(args.sr)
    os.makedirs(args.out)
    wavPath = args.wav
    outPath = args.out

    assert args.sr == 16000 or args.sr == 48000

    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
            os.makedirs(f"./{outPath}/{spks}")
            print(f">>>>>>>>>>{spks}<<<<<<<<<<")
            for file in os.listdir(f"./{wavPath}/{spks}"):
                if file.endswith(".wav"):
                    file = file[:-4]
                    resample_wave(f"{wavPath}/{spks}/{file}.wav", f"{outPath}/{spks}/{file}.wav", args.sr)
        else:
            file = spks
            if file.endswith(".wav"):
                file = file[:-4]
                resample_wave(f"{wavPath}/{file}.wav", f"{outPath}/{file}.wav", args.sr)
