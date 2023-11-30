import os
import librosa
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.io import wavfile


def resample_wave(wav_in, wav_out, sample_rate):
    wav, _ = librosa.load(wav_in, sr=sample_rate)
    wav = wav / np.abs(wav).max() * 0.6
    wav = wav / max(0.01, np.max(np.abs(wav))) * 32767 * 0.6
    wavfile.write(wav_out, sample_rate, wav.astype(np.int16))


def process_file(file, wavPath, spks, outPath, sr):
    if file.endswith(".wav"):
        file = file[:-4]
        resample_wave(f"{wavPath}/{spks}/{file}.wav", f"{outPath}/{spks}/{file}.wav", sr)


def process_files_with_thread_pool(wavPath, spks, outPath, sr, thread_num=None):
    files = [f for f in os.listdir(f"./{wavPath}/{spks}") if f.endswith(".wav")]

    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        futures = {executor.submit(process_file, file, wavPath, spks, outPath, sr): file for file in files}

        for future in tqdm(as_completed(futures), total=len(futures), desc=f'Processing {sr} {spks}'):
            future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav", help="wav", dest="wav", required=True)
    parser.add_argument("-o", "--out", help="out", dest="out", required=True)
    parser.add_argument("-s", "--sr", help="sample rate", dest="sr", type=int, required=True)
    parser.add_argument("-t", "--thread_count", help="thread count to process, set 0 to use all cpu cores", dest="thread_count", type=int, default=1)

    args = parser.parse_args()
    print(args.wav)
    print(args.out)
    print(args.sr)

    os.makedirs(args.out, exist_ok=True)
    wavPath = args.wav
    outPath = args.out

    assert args.sr == 16000 or args.sr == 32000

    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
            os.makedirs(f"./{outPath}/{spks}", exist_ok=True)
            if args.thread_count == 0:
                process_num = os.cpu_count() // 2 + 1
            else:
                process_num = args.thread_count
            process_files_with_thread_pool(wavPath, spks, outPath, args.sr, process_num)
