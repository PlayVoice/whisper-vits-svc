import os
import argparse
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("dataset_speaker", type=str)
    parser.add_argument("dataset_singer", type=str)
    data_speaker = parser.parse_args().dataset_speaker
    data_singer = parser.parse_args().dataset_singer

    os.makedirs(data_singer)

    for speaker in os.listdir(data_speaker):
        print(speaker)
        subfile_num = 0
        speaker_ave = 0
        for file in os.listdir(os.path.join(data_speaker, speaker)):
            if file.endswith(".npy"):
                source_embed = np.load(
                    os.path.join(data_speaker, speaker, file))
                source_embed = source_embed.astype(np.float32)
                speaker_ave = speaker_ave + source_embed
                subfile_num = subfile_num + 1
        speaker_ave = speaker_ave / subfile_num

        np.save(os.path.join(data_singer, f"{speaker}.spk.npy"),
                speaker_ave, allow_pickle=False)
