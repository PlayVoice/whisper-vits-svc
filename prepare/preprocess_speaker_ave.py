import os
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def load_embed_file(file):
    if file.endswith(".npy"):
        source_embed = np.load(
            os.path.join(data_speaker, speaker, file))
        source_embed = source_embed.astype(np.float32)
        return source_embed
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("dataset_speaker", type=str)
    parser.add_argument("dataset_singer", type=str)
    parser.add_argument("-t", "--thread_count", help="thread count to process, set 0 to use all cpu cores", dest="thread_count", type=int, default=1)
    
    data_speaker = parser.parse_args().dataset_speaker
    data_singer = parser.parse_args().dataset_singer
    thread_count = parser.parse_args().thread_count

    if not os.path.exists(data_singer):
        os.makedirs(data_singer)

    for speaker in os.listdir(data_speaker):
        print(speaker)
        subfile_num = 0
        speaker_ave = 0
        if thread_count == 0:
            process_num = os.cpu_count()
        else:
            process_num = thread_count
            
        with ThreadPoolExecutor(max_workers=10) as executor:
            for file in tqdm(os.listdir(os.path.join(data_speaker, speaker))):
                future = executor.submit(load_embed_file, file)
                source_embed = future.result()
                if source_embed is not None:
                    speaker_ave = speaker_ave + source_embed
                    subfile_num = subfile_num + 1
        if subfile_num == 0:
            continue
        speaker_ave = speaker_ave / subfile_num

        np.save(os.path.join(data_singer, f"{speaker}.spk.npy"),
                speaker_ave, allow_pickle=False)
