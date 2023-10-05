import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import librosa
import argparse
import numpy as np
import parselmouth


def compute_f0_mouth(path, device):
    # pip install praat-parselmouth
    x, sr = librosa.load(path, sr=16000)
    assert sr == 16000
    lpad = 1024 // 160
    rpad = lpad
    f0 = parselmouth.Sound(x, sr).to_pitch_ac(
        time_step=160 / sr,
        voicing_threshold=0.5,
        pitch_floor=30,
        pitch_ceiling=1000).selected_array['frequency']
    f0 = np.pad(f0, [[lpad, rpad]], mode='constant')
    return f0


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
    parser.add_argument("-w", "--wav", help="wav", dest="wav", required=True)
    parser.add_argument("-p", "--pit", help="pit", dest="pit", required=True)  # csv for excel
    args = parser.parse_args()
    print(args.wav)
    print(args.pit)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pitch = compute_f0_mouth(args.wav, device)
    save_csv_pitch(pitch, args.pit)
    # tmp = load_csv_pitch(args.pit)
    # save_csv_pitch(tmp, "tmp.csv")
