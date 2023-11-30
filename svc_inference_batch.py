import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tqdm
import torch
import argparse

from whisper.inference import load_model, pred_ppg

# How to use
# python svc_inference_batch.py --config configs/base.yaml --model vits_pretrain/sovits5.0.pth --wave test_waves/ --spk configs/singers/singer0047.npy

out_path = "./_svc_out"
os.makedirs(out_path, exist_ok=True)

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
    parser.add_argument('--shift', type=int, default=0,
                    help="Pitch shift key.")
    args = parser.parse_args()
    wave_path = args.wave
    assert os.path.isdir(wave_path), f"{wave_path} is not folder"
    waves = [file for file in os.listdir(wave_path) if file.endswith(".wav")]
    for file in waves:
        print(file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    whisper = load_model(os.path.join("whisper_pretrain", "large-v2.pt"), device=device)
    for file in tqdm.tqdm(waves, desc="whisper"):
        pred_ppg(whisper, f"{wave_path}/{file}", f"{out_path}/{file}.ppg.npy", device=device)
    del whisper

    for file in tqdm.tqdm(waves, desc="svc"):
        os.system(
            f"python svc_inference.py --config {args.config} --model {args.model} --wave {wave_path}/{file} --ppg {out_path}/{file}.ppg.npy --spk {args.spk} --shift {args.shift}")
        os.system(f"mv svc_out.wav {out_path}/{file}")
        os.system(f"rm {out_path}/{file}.ppg.npy")
