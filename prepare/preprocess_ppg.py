import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
import numpy as np
import argparse
import torch
from tqdm import tqdm
from multiprocessing import Pool
from whisper.model import Whisper, ModelDimensions
from whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram
from concurrent.futures import ThreadPoolExecutor, as_completed


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
    ppgln = audln // 320
    audio = pad_or_trim(audio)
    mel = log_mel_spectrogram(audio).half().to(whisper.device)
    with torch.no_grad():
        ppg = whisper.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
        ppg = ppg[:ppgln,] # [length, dim=1024]
        # print(ppg.shape)
        np.save(ppgPath, ppg, allow_pickle=False)

def process_file(file):
    if file.endswith(".wav"):
        file = file[:-4]
        pred_ppg(whisper, f"{wavPath}/{spks}/{file}.wav", f"{ppgPath}/{spks}/{file}.ppg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("-w", "--wav", help="wav", dest="wav")
    parser.add_argument("-p", "--ppg", help="ppg", dest="ppg")
    parser.add_argument("-t", "--thread_count", help="thread count to process, set 0 to use all cpu cores", dest="thread_count", type=int, default=1)
    
    args = parser.parse_args()
    print(args.wav)
    print(args.ppg)
    if not os.path.exists(args.ppg):
        os.makedirs(args.ppg)

    wavPath = args.wav
    ppgPath = args.ppg

    whisper = load_model(os.path.join("whisper_pretrain", "large-v2.pt"))

    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
            if not os.path.exists(f"./{ppgPath}/{spks}"):
                os.makedirs(f"./{ppgPath}/{spks}")
            print(f">>>>>>>>>>{spks}<<<<<<<<<<")
            if args.thread_count == 1:
                for file in os.listdir(f"./{wavPath}/{spks}"):
                    if file.endswith(".wav"):
                        print(file)
                        file = file[:-4]
                        pred_ppg(whisper, f"{wavPath}/{spks}/{file}.wav", f"{ppgPath}/{spks}/{file}.ppg")
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
