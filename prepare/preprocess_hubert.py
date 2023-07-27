import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import argparse
import torch
import librosa

from tqdm import tqdm
from multiprocessing import Pool
from hubert import hubert_model
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_audio(file: str, sr: int = 16000):
    x, sr = librosa.load(file, sr=sr)
    return x


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


def process_file(file):
    if file.endswith(".wav"):
        file = file[:-4]
        pred_vec(hubert, f"{wavPath}/{spks}/{file}.wav", f"{vecPath}/{spks}/{file}.vec", device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("-w", "--wav", help="wav", dest="wav")
    parser.add_argument("-v", "--vec", help="vec", dest="vec")
    parser.add_argument("-t", "--thread_count", help="thread count to process, set 0 to use all cpu cores", dest="thread_count", type=int, default=1)
    
    args = parser.parse_args()
    print(args.wav)
    print(args.vec)
    os.makedirs(args.vec, exist_ok=True)

    wavPath = args.wav
    vecPath = args.vec

    assert torch.cuda.is_available()
    device = "cuda"
    hubert = load_model(os.path.join(
        "hubert_pretrain", "hubert-soft-0d54a1f4.pt"), device)

    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
            os.makedirs(f"./{vecPath}/{spks}", exist_ok=True)
            print(f">>>>>>>>>>{spks}<<<<<<<<<<")
            if args.thread_count == 1:
                for file in os.listdir(f"./{wavPath}/{spks}"):
                    if file.endswith(".wav"):
                        print(file)
                        file = file[:-4]
                        pred_vec(hubert, f"{wavPath}/{spks}/{file}.wav", f"{vecPath}/{spks}/{file}.vec", device)
            else:
                if args.thread_count == 0:
                    process_num = os.cpu_count()
                else:
                    process_num = args.thread_count
                with ThreadPoolExecutor(max_workers=process_num) as executor:
                    futures = [executor.submit(process_file, file) for file in os.listdir(f"./{wavPath}/{spks}")]
                    for future in tqdm(as_completed(futures), total=len(futures)):
                        pass
                # with Pool(processes=process_num) as pool:
                #     results = [pool.apply_async(process_file, (file,)) for file in os.listdir(f"./{wavPath}/{spks}")]
                #     for result in tqdm(results, total=len(results)):
                #         result.wait()
