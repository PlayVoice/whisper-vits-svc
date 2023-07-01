import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import argparse
import torch

from whisper.audio import load_audio
from hubert import hubert_model


def load_model(path, device):
    model = hubert_model.hubert_soft(path)
    model.eval()
    model.half()
    model.to(device)
    return model


def pred_vec(model, wavPath, vecPath, device):
    feats = load_audio(wavPath)
    feats = torch.from_numpy(feats).to(device)
    feats = feats[None, None, :].half()
    with torch.no_grad():
        vec = model.units(feats).squeeze().data.cpu().float().numpy()
        # print(vec.shape)   # [length, dim=256] hop=320
        np.save(vecPath, vec, allow_pickle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("-w", "--wav", help="wav", dest="wav")
    parser.add_argument("-v", "--vec", help="vec", dest="vec")
    args = parser.parse_args()
    print(args.wav)
    print(args.vec)

    wavPath = args.wav
    vecPath = args.vec

    assert torch.cuda.is_available()
    device = "cuda"
    hubert = load_model(os.path.join(
        "hubert_pretrain", "hubert-soft-0d54a1f4.pt"), device)
    pred_vec(hubert, wavPath, vecPath, device)
