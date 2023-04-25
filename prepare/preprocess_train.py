import os
import random


if __name__ == "__main__":
    os.makedirs("./files/", exist_ok=True)

    rootPath = "./data_svc/waves-48k/"
    all_items = []
    for spks in os.listdir(f"./{rootPath}"):
        if os.path.isdir(f"./{rootPath}/{spks}"):
            for file in os.listdir(f"./{rootPath}/{spks}"):
                if file.endswith(".wav"):
                    file = file[:-4]
                    path_spk = f"./data_svc/speaker/{spks}/{file}.spk.npy"
                    path_wave = f"./data_svc/waves-48k/{spks}/{file}.wav"
                    path_spec = f"./data_svc/specs/{spks}/{file}.pt"
                    path_pitch = f"./data_svc/pitch/{spks}/{file}.pit.npy"
                    path_whisper = f"./data_svc/whisper/{spks}/{file}.ppg.npy"
                    assert os.path.isfile(path_spk), path_spk
                    assert os.path.isfile(path_wave), path_wave
                    assert os.path.isfile(path_spec), path_spec
                    assert os.path.isfile(path_pitch), path_pitch
                    assert os.path.isfile(path_whisper), path_whisper
                    all_items.append(
                        f"{path_wave}|{path_spec}|{path_pitch}|{path_whisper}|{path_spk}")
        else:
            file = spks
            if file.endswith(".wav"):
                file = file[:-4]
                path_spk = f"./data_svc/speaker/{file}.spk.npy"
                path_wave = f"./data_svc/waves-48k/{file}.wav"
                path_spec = f"./data_svc/specs/{file}.pt"
                path_pitch = f"./data_svc/pitch/{file}.pit.npy"
                path_whisper = f"./data_svc/whisper/{file}.ppg.npy"
                assert os.path.isfile(path_spk), path_spk
                assert os.path.isfile(path_wave), path_wave
                assert os.path.isfile(path_spec), path_spec
                assert os.path.isfile(path_pitch), path_pitch
                assert os.path.isfile(path_whisper), path_whisper
                all_items.append(
                    f"{path_wave}|{path_spec}|{path_pitch}|{path_whisper}|{path_spk}")

    random.shuffle(all_items)
    valids = all_items[:50]
    valids.sort()
    trains = all_items[50:]
    # trains.sort()
    fw = open("./files/valid.txt", "w", encoding="utf-8")
    for strs in valids:
        print(strs, file=fw)
    fw.close()
    fw = open("./files/train.txt", "w", encoding="utf-8")
    for strs in trains:
        print(strs, file=fw)
    fw.close()
