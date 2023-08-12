import os
import torch
import argparse
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_speaker", type=str)
    parser.add_argument("dataset_singer", type=str)

    data_speaker = parser.parse_args().dataset_speaker
    data_singer = parser.parse_args().dataset_singer

    os.makedirs(data_singer, exist_ok=True)

    for speaker in os.listdir(data_speaker):
        subfile_num = 0
        speaker_ave = 0

        for file in tqdm(os.listdir(os.path.join(data_speaker, speaker)), desc=f"average {speaker}"):
            if not file.endswith(".npy"):
                continue
            source_embed = np.load(os.path.join(data_speaker, speaker, file))
            source_embed = source_embed.astype(np.float32)
            speaker_ave = speaker_ave + source_embed
            subfile_num = subfile_num + 1
        if subfile_num == 0:
            continue
        speaker_ave = speaker_ave / subfile_num

        np.save(os.path.join(data_singer, f"{speaker}.spk.npy"),
                speaker_ave, allow_pickle=False)

        # rewrite timbre code by average, if similarity is larger than cmp_val
        rewrite_timbre_code = False
        if not rewrite_timbre_code:
            continue
        cmp_src = torch.FloatTensor(speaker_ave)
        cmp_num = 0
        cmp_val = 0.85
        for file in tqdm(os.listdir(os.path.join(data_speaker, speaker)), desc=f"rewrite {speaker}"):
            if not file.endswith(".npy"):
                continue
            cmp_tmp = np.load(os.path.join(data_speaker, speaker, file))
            cmp_tmp = cmp_tmp.astype(np.float32)
            cmp_tmp = torch.FloatTensor(cmp_tmp)
            cmp_cos = torch.cosine_similarity(cmp_src, cmp_tmp, dim=0)
            if (cmp_cos > cmp_val):
                cmp_num += 1
                np.save(os.path.join(data_speaker, speaker, file),
                        speaker_ave, allow_pickle=False)
        print(f"rewrite timbre for {speaker} with :", cmp_num)
