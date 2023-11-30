import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import argparse
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from vits import spectrogram
from vits import utils
from omegaconf import OmegaConf


def compute_spec(hps, filename, specname):
    audio, sampling_rate = utils.load_wav_to_torch(filename)
    assert sampling_rate == hps.sampling_rate, f"{sampling_rate} is not {hps.sampling_rate}"
    audio_norm = audio / hps.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    n_fft = hps.filter_length
    sampling_rate = hps.sampling_rate
    hop_size = hps.hop_length
    win_size = hps.win_length
    spec = spectrogram.spectrogram_torch(
        audio_norm, n_fft, sampling_rate, hop_size, win_size, center=False)
    spec = torch.squeeze(spec, 0)
    torch.save(spec, specname)


def process_file(file):
    if file.endswith(".wav"):
        file = file[:-4]
        compute_spec(hps.data, f"{wavPath}/{spks}/{file}.wav", f"{spePath}/{spks}/{file}.pt")


def process_files_with_thread_pool(wavPath, spks, thread_num):
    files = os.listdir(f"./{wavPath}/{spks}")
    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        list(tqdm(executor.map(process_file, files), total=len(files), desc=f'Processing spec {spks}'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav", help="wav", dest="wav", required=True)
    parser.add_argument("-s", "--spe", help="spe", dest="spe", required=True)
    parser.add_argument("-t", "--thread_count", help="thread count to process, set 0 to use all cpu cores", dest="thread_count", type=int, default=1)

    args = parser.parse_args()
    print(args.wav)
    print(args.spe)

    os.makedirs(args.spe, exist_ok=True)
    wavPath = args.wav
    spePath = args.spe
    hps = OmegaConf.load("./configs/base.yaml")

    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
            os.makedirs(f"./{spePath}/{spks}", exist_ok=True)
            if args.thread_count == 0:
                process_num = os.cpu_count() // 2 + 1
            else:
                process_num = args.thread_count
            process_files_with_thread_pool(wavPath, spks, process_num)
