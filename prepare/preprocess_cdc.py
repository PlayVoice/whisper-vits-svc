import os
import argparse
import torch
import torchaudio

from tqdm import tqdm
from scipy.io.wavfile import read
from scipy.io.wavfile import write
# torch=1.9.0 ->  pip install torchaudio==0.9.0 -i https://mirrors.aliyun.com/pypi/simple/
# this file is for VCTK


MAX_WAV_VALUE = 32768.0


def cut_direct_content(iWave, oWave):
    source, sr = torchaudio.load(iWave)
    stft = torch.stft(source, 1024, 256, 1024, torch.hann_window(1024), return_complex=True)
    stft[:, 0, :] = 0
    stft[:, 1, :] = 0
    istft = torch.istft(stft, 1024, 256, 1024, torch.hann_window(1024))
    audio = istft.squeeze()
    audio = MAX_WAV_VALUE * audio
    audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
    audio = audio.short()
    audio = audio.data.cpu().detach().numpy()
    write(oWave, sr, audio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="input path", dest="inPath", required=True)
    parser.add_argument("-o", help="output path", dest="outPath", required=True)

    args = parser.parse_args()
    print(args.inPath)
    print(args.outPath)

    os.makedirs(args.outPath, exist_ok=True)
    rootPath = args.inPath
    outPath = args.outPath

    for spks in os.listdir(rootPath):
        if (os.path.isdir(f"./{rootPath}/{spks}")):
            os.makedirs(f"./{outPath}/{spks}", exist_ok=True)

            files = [f for f in os.listdir(f"./{rootPath}/{spks}") if f.endswith(".wav")]
            for file in tqdm(files, desc=f'Processing cdc {spks}'):
                iWave = f"./{rootPath}/{spks}/{file}"
                oWave = f"./{outPath}/{spks}/{file}"
                cut_direct_content(iWave, oWave)
