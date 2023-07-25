import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import librosa
import argparse
import numpy as np
import crepe


def compute_f0_voice(filename, device):
    audio, sr = librosa.load(filename, sr=16000)
    assert sr == 16000
    audio = torch.tensor(np.copy(audio))[None]
    audio = audio + torch.randn_like(audio) * 0.001
    # Here we'll use a 10 millisecond hop length
    hop_length = 160
    fmin = 50
    fmax = 1000
    model = "full"
    batch_size = 512
    pitch = crepe.predict(
        audio,
        sr,
        hop_length,
        fmin,
        fmax,
        model,
        batch_size=batch_size,
        device=device,
        return_periodicity=False,
    )
    pitch = crepe.filter.mean(pitch, 5)
    pitch = pitch.squeeze(0)
    return pitch


def compute_f0_sing(filename, device):
    audio, sr = librosa.load(filename, sr=16000)
    assert sr == 16000
    audio = torch.tensor(np.copy(audio))[None]
    audio = audio + torch.randn_like(audio) * 0.001
    # Here we'll use a 20 millisecond hop length
    hop_length = 320
    fmin = 50
    fmax = 1000
    model = "full"
    batch_size = 512
    pitch = crepe.predict(
        audio,
        sr,
        hop_length,
        fmin,
        fmax,
        model,
        batch_size=batch_size,
        device=device,
        return_periodicity=False,
    )
    pitch = np.repeat(pitch, 2, -1)  # 320 -> 160 * 2
    pitch = crepe.filter.mean(pitch, 5)
    pitch = pitch.squeeze(0)
    return pitch


def save_csv_pitch(pitch, path):
    with open(path, "w", encoding='utf-8') as pitch_file:
        for i in range(len(pitch)):
            t = i * 10
            minute = t // 60000
            seconds = (t - minute * 60000) // 1000
            millisecond = t % 1000
            print(
                f"{minute}m {seconds}s {millisecond:3d},{int(pitch[i])}", file=pitch_file)


def load_csv_pitch(path):
    pitch = []
    with open(path, "r", encoding='utf-8') as pitch_file:
        for line in pitch_file.readlines():
            pit = line.strip().split(",")[-1]
            pitch.append(int(pit))
    return pitch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("-w", "--wav", help="wav", dest="wav")
    parser.add_argument("-p", "--pit", help="pit", dest="pit")  # csv for excel
    args = parser.parse_args()
    print(args.wav)
    print(args.pit)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pitch = compute_f0_sing(args.wav, device)
    save_csv_pitch(pitch, args.pit)
    #tmp = load_csv_pitch(args.pit)
    #save_csv_pitch(tmp, "tmp.csv")
