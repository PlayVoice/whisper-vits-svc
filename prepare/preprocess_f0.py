import os
import numpy as np
import librosa
import pyworld
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def compute_f0(path, save):
    x, sr = librosa.load(path, sr=16000)
    assert sr == 16000
    f0, t = pyworld.dio(
        x.astype(np.double),
        fs=sr,
        f0_ceil=900,
        frame_period=1000 * 160 / sr,
    )
    f0 = pyworld.stonemask(x.astype(np.double), f0, t, fs=16000)
    for index, pitch in enumerate(f0):
        f0[index] = round(pitch, 1)
    np.save(save, f0, allow_pickle=False)


def process_file(file, wavPath, spks, pitPath):
    if file.endswith(".wav"):
        file = file[:-4]
        compute_f0(f"{wavPath}/{spks}/{file}.wav", f"{pitPath}/{spks}/{file}.pit")


def process_files_with_process_pool(wavPath, spks, pitPath, process_num=None):
    files = [f for f in os.listdir(f"./{wavPath}/{spks}") if f.endswith(".wav")]

    with ProcessPoolExecutor(max_workers=process_num) as executor:
        futures = {executor.submit(process_file, file, wavPath, spks, pitPath): file for file in files}

        for future in tqdm(as_completed(futures), total=len(futures), desc=f'Processing f0 {spks}'):
            future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav", help="wav", dest="wav", required=True)
    parser.add_argument("-p", "--pit", help="pit", dest="pit", required=True)
    parser.add_argument("-t", "--thread_count", help="thread count to process, set 0 to use all cpu cores", dest="thread_count", type=int, default=1)
    
    args = parser.parse_args()
    print(args.wav)
    print(args.pit)

    os.makedirs(args.pit, exist_ok=True)
    wavPath = args.wav
    pitPath = args.pit

    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
            os.makedirs(f"./{pitPath}/{spks}", exist_ok=True)
            if args.thread_count == 0:
                process_num = os.cpu_count() // 2 + 1
            else:
                process_num = args.thread_count
            process_files_with_process_pool(wavPath, spks, pitPath, process_num)
