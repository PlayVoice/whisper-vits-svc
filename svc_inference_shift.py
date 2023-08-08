import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import argparse
import numpy as np

from omegaconf import OmegaConf
from scipy.io.wavfile import write
from pitch import load_csv_pitch
from vits.models import SynthesizerInfer
from svc_inference import load_svc_model, svc_infer


def main(args):
    if (args.ppg == None):
        args.ppg = "svc_tmp.ppg.npy"
        print(
            f"Auto run : python whisper/inference.py -w {args.wave} -p {args.ppg}")
        os.system(f"python whisper/inference.py -w {args.wave} -p {args.ppg}")

    if (args.vec == None):
        args.vec = "svc_tmp.vec.npy"
        print(
            f"Auto run : python hubert/inference.py -w {args.wave} -v {args.vec}")
        os.system(f"python hubert/inference.py -w {args.wave} -v {args.vec}")

    if (args.pit == None):
        args.pit = "svc_tmp.pit.csv"
        print(
            f"Auto run : python pitch/inference.py -w {args.wave} -p {args.pit}")
        os.system(f"python pitch/inference.py -w {args.wave} -p {args.pit}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = OmegaConf.load(args.config)
    model = SynthesizerInfer(
        hp.data.filter_length // 2 + 1,
        hp.data.segment_size // hp.data.hop_length,
        hp)
    load_svc_model(args.model, model)
    model.eval()
    model.to(device)

    spk = np.load(args.spk)
    spk = torch.FloatTensor(spk)

    ppg = np.load(args.ppg)
    ppg = np.repeat(ppg, 2, 0)
    ppg = torch.FloatTensor(ppg)

    vec = np.load(args.vec)
    vec = np.repeat(vec, 2, 0)
    vec = torch.FloatTensor(vec)

    pit = load_csv_pitch(args.pit)

    shift_l = args.shift_l
    shift_r = args.shift_r

    print(f"pitch shift: [{shift_l}, {shift_r}]")

    for shift in range(shift_l, shift_r + 1):
        print(shift)
        tmp = np.array(pit)
        tmp = tmp * (2 ** (shift / 12))
        tmp = torch.FloatTensor(tmp)

        out_audio = svc_infer(model, spk, tmp, ppg, vec, hp, device)
        write(os.path.join("./_svc_out", f"svc_out_{shift}.wav"),
              hp.data.sampling_rate, out_audio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help="yaml file for config.")
    parser.add_argument('--model', type=str, required=True,
                        help="path of model for evaluation")
    parser.add_argument('--wave', type=str, required=True,
                        help="Path of raw audio.")
    parser.add_argument('--spk', type=str, required=True,
                        help="Path of speaker.")
    parser.add_argument('--ppg', type=str,
                        help="Path of content vector.")
    parser.add_argument('--vec', type=str,
                        help="Path of hubert vector.")
    parser.add_argument('--pit', type=str,
                        help="Path of pitch csv file.")
    parser.add_argument('--shift_l', type=int, default=0,
                        help="Pitch shift key for [shift_l, shift_r]")
    parser.add_argument('--shift_r', type=int, default=0,
                        help="Pitch shift key for [shift_l, shift_r]")
    args = parser.parse_args()

    assert args.shift_l >= -12
    assert args.shift_r >= -12
    assert args.shift_l <= 12
    assert args.shift_r <= 12
    assert args.shift_l <= args.shift_r

    os.makedirs("./_svc_out", exist_ok=True)

    main(args)
