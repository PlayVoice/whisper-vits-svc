import os
import argparse

from tqdm import tqdm
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub import effects
# this file is for VCTK, use after CDC


def trim_silence(iWave, oWave):
    try:
        audio = AudioSegment.from_wav(iWave)
        # audio = effects.normalize(audio, 6)# max - 6dB
        audio_chunks = split_on_silence(
            audio,
            min_silence_len=200,
            silence_thresh=-45,
            keep_silence=200,
        )
        for chunk in audio_chunks[1:]:
            audio_chunks[0] += chunk
        audio_chunks[0].export(oWave, format="wav")
    except Exception as e:
        print(str(e))
        print(iWave)


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
            for file in tqdm(files, desc=f'Processing sil {spks}'):
                iWave = f"./{rootPath}/{spks}/{file}"
                oWave = f"./{outPath}/{spks}/{file}"
                trim_silence(iWave, oWave)
